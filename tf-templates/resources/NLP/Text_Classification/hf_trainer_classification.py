# !pip install transformers datasets evaluate accelerate

from datasets import load_dataset
import numpy as np
import evaluate

imdb = load_dataset("imdb")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_imdb = imdb.map(preprocess_function, batched=True)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="distilbert",
    learning_rate=2e-5,
    optim="adamw_torch", #adamw_torch_fused
    per_device_train_batch_size=32,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()