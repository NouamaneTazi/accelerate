import math

import torch
from torch.utils.data import DataLoader

import evaluate
from accelerate import Accelerator
from datasets import load_dataset
from fastcore.script import call_parse
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_dataloader(accelerator: Accelerator, drop_last=False):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = load_dataset("glue", "mrpc", split="validation")

    def tokenize_function(examples):
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    return DataLoader(tokenized_datasets, shuffle=False, collate_fn=collate_fn, batch_size=16, drop_last=drop_last)


def get_setup(dispatch_batches, split_batches, drop_last):
    accelerator = Accelerator(dispatch_batches=dispatch_batches, split_batches=split_batches)
    dataloader = get_dataloader(accelerator, drop_last)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)
    model.load_state_dict(torch.load("../nlp_example/pytorch_model.bin"))
    ddp_model, ddp_dataloader = accelerator.prepare(model, dataloader)
    return {"ddp": [ddp_model, ddp_dataloader, "cuda:0"], "no": [model, dataloader, accelerator.device]}, accelerator


@call_parse
def main(
    dispatch_batches: bool = False,  # Whether to dispatch batches
    split_batches: bool = False,  # Whether to split batches
):
    drop_last = False if not dispatch_batches else True
    metric = evaluate.load("glue", "mrpc")
    setup, accelerator = get_setup(dispatch_batches, split_batches, drop_last)
    # First do baseline
    if accelerator.is_local_main_process:
        print("Running baseline")
    model, dataloader, device = setup["no"]
    if accelerator.is_local_main_process:
        print(f"Len dl: {len(dataloader)}\nLen dset: {len(dataloader.dataset)}\n")
    model.to(device)
    model.eval()
    for batch in dataloader:
        batch.to(device)
        with torch.inference_mode():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=preds, references=batch["labels"])
    baseline = metric.compute()

    # Then do distributed
    if accelerator.is_local_main_process:
        print("Running with basic distributed setup")
    model, dataloader, device = setup["ddp"]
    model.eval()
    samples_seen = 0
    for step, batch in enumerate(dataloader):
        with torch.inference_mode():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        references = batch["labels"]
        preds, references = accelerator.gather((preds, references))
        # This should be able to run with dispatch_batches and the same code
        # For now passes if using `if not dispatch_batches` and running this script with `split_batches`
        if not dispatch_batches:
            samples_seen += references.shape[0]
            if step == (len(dataloader) - 1):
                preds = preds[: len(dataloader.dataset) - samples_seen]
                references = references[: len(dataloader.dataset) - samples_seen]
        metric.add_batch(predictions=preds, references=references)
    distributed = metric.compute()

    # Then do distributed with gradient state
    if accelerator.is_local_main_process:
        print("Running with Gradient State")
    model, dataloader, device = setup["ddp"]
    model.eval()
    for batch in dataloader:
        with torch.inference_mode():
            outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        references = batch["labels"]
        preds, references = accelerator.gather((preds, references))
        if accelerator.gradient_state.end_of_dataloader and not accelerator.gradient_state.dispatch_batches:
            preds = preds[: accelerator.gradient_state.remainder]
            references = references[: accelerator.gradient_state.remainder]
        metric.add_batch(predictions=preds, references=references)
    gradient_state = metric.compute()

    for key in "accuracy f1".split():
        if not math.isclose(baseline[key], distributed[key]) and accelerator.is_local_main_process:
            print(
                f"Baseline and Distributed are not the same for key {key}:\n\tBaseline: {baseline[key]}\n\tDistributed: {distributed[key]}\n"
            )
        if not math.isclose(baseline[key], gradient_state[key]) and accelerator.is_local_main_process:
            print(
                f"Baseline and Gradient State are not the same for key {key}:\n\tBaseline: {baseline[key]}\n\tState: {gradient_state[key]}\n"
            )
