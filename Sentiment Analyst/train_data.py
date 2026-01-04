# 02_train_model.py — small-data, CPU-friendly full fine-tune
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch, os as _os

set_seed(42)

# Load tokenized splits
train_ds = load_from_disk("train_data")
val_ds   = load_from_disk("val_data")

# Cap sizes to keep runtime short on CPU (raise these if you have time/GPU)
def cap(ds, n):
    return ds.shuffle(seed=42).select(range(min(len(ds), n)))

train_ds = cap(train_ds, 6000)   
val_ds   = cap(val_ds,   1500)

# Model choice — tiny for speed;
MODEL_NAME = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Ensure readable labels are saved with the model
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Dynamic padding speeds things up
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Minimal args that work on older transformers; disable WANDB cleanly via report_to="none"
# Some older versions don't accept report_to; guard it.
def make_args():
    base = dict(
        output_dir="sentiment-model", 
        per_device_train_batch_size=16,   # drop to 8 on CPU if OOM
        per_device_eval_batch_size=32,
        num_train_epochs=8,               # small data needs more epochs
        learning_rate=2e-5,               # good for full fine-tune
        logging_steps=20,
        seed=42,
    )
    try:
        return TrainingArguments(**base, report_to="none")
    except TypeError:
        return TrainingArguments(**base)

args = make_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,           # fine for v4.x; v5 will use processing_class
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train and then explicitly evaluate (compatible with older versions)
trainer.train()
results = trainer.evaluate(eval_dataset=val_ds)
print("📈 Validation:", results)

# Save model + tokenizer (with label names)
trainer.save_model("sentiment-model")
tokenizer.save_pretrained("sentiment-model")
print("✅ Training complete. Saved to ./sentiment-model")