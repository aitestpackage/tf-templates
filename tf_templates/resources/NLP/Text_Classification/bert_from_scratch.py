# https://openreview.net/pdf?id=PdaPky8MUn
# Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
config = BertConfig(
    vocab_size=30522,
    hidden_size=384,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=768,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    classifier_dropout=0.1,
)
model = BertForMaskedLM(config)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.30
)

from transformers import Trainer, TrainingArguments
from datasets import load_dataset
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
dataset = dataset.map(preprocess_function, batched=True)
training_args = TrainingArguments(
    output_dir="trainingbert",
    learning_rate=2e-5,
    optim="adamw_torch", #adamw_torch_fused
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    warmup_steps=10,
    eval_strategy="no",
    save_strategy="best",
    logging_steps=10,
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
    label_smoothing_factor=0.01,
    fp16=True,
    fp16_opt_level="O3",
    # bf16=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()

import numpy as np
from datasets import load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction,
    BertConfig,
)
import evaluate
# === Load Dataset (IMDb for binary classification) ===
dataset = load_dataset("imdb")

# === Tokenize the Dataset ===
def preprocess_function(examples):
    encoding = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    encoding["labels"] = examples["label"]
    return encoding

tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=4)

# === Load Model with Classification Head ===
from transformers import BertForSequenceClassification

# Suppose `mlm_model` is your trained BertForMaskedLM in memory
cfg = model.config
cfg.num_labels = 2       # e.g. 2 for binary
cls_model = BertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=None,   # no local folder
    config=cfg,
    state_dict=model.state_dict(),    # load these weights
    ignore_mismatched_sizes=True,
)

# === Data Collator for Padding ===
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# === Define Evaluation Metrics ===
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_metric.compute(predictions=preds, references=p.label_ids)
    f1 = f1_metric.compute(predictions=preds, references=p.label_ids)
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="bert-classifier",
    learning_rate=3e-4,
    optim="adamw_torch", #adamw_torch_fused
    per_device_train_batch_size=256,
    per_device_eval_batch_size=1024,
    num_train_epochs=2,
    weight_decay=0.01,
    warmup_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="best",
    logging_steps=10,
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
    label_smoothing_factor=0.01,
    fp16=True,
    fp16_opt_level="O3",
    # bf16=True,
)
# === Trainer ===
trainer = Trainer(
    model=cls_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# === Train ===
trainer.train()
# === Save the Fine-Tuned Model ===
trainer.save_model("bert-classifier")

# === Inference Example (Optional) ===
from transformers import pipeline

clf = pipeline("text-classification", model="bert-classifier", tokenizer=tokenizer)
print(clf("This movie was surprisingly good!"))