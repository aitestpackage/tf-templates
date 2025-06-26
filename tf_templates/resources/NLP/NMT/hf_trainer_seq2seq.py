# !pip install transformers datasets evaluate sacrebleu

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np

# 1. Load & split the dataset
raw_datasets = load_dataset("kde4", lang1="en", lang2="fr", trust_remote_code=True)
split = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
split["validation"] = split.pop("test")

# 2. Tokenizer & preprocessing
checkpoint = "google-t5/t5-small"
tokenizer  = AutoTokenizer.from_pretrained(checkpoint)

max_length = 128
def preprocess_function(examples):
    inputs  = [t["en"] for t in examples["translation"]]
    targets = [t["fr"] for t in examples["translation"]]
    # This produces input_ids, attention_mask, AND labels
    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    return model_inputs

tokenized = split.map(preprocess_function, batched=True, num_proc=4)

# 3. Metrics
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds  = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):  # some models return (logits, past_key_values)
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 with pad_token_id for correct decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    bleu = metric.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    gen_len = np.mean([np.count_nonzero(p != tokenizer.pad_token_id) for p in preds])

    return {"bleu": round(bleu, 4), "gen_len": round(gen_len, 4)}

# 4. Model & Data Collator
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="distilbert",
    learning_rate=2e-5,
    optim="adamw_torch", #adamw_torch_fused
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
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

# 6. Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. Start training
trainer.train()