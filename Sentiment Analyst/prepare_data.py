# 01_prepare_data.py
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# If in Colab, let user upload
try:
    from google.colab import files  # type: ignore
    print("📁 Upload your IMDb CSV (e.g., IMDB Dataset.csv)")
    files.upload()
except Exception:
    pass

# Find CSV with common names
candidates = [
    "IMDB Dataset.csv", "IMDB_Dataset.csv", "IMDB Dateset.csv",
    "imdb_dataset.csv", "imdb.csv"
]
csv_path = next((p for p in candidates if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("Could not find CSV. Upload e.g. 'IMDB Dataset.csv'.")

print(f"✔ Loading {csv_path}")
df = pd.read_csv(csv_path)

# Normalize columns
cols = {c.lower().strip(): c for c in df.columns}
text_col = cols.get("text") or cols.get("review")
label_col = cols.get("label") or cols.get("sentiment")
if text_col is None or label_col is None:
    raise ValueError("CSV must contain 'text' (or 'review') and 'label' (or 'sentiment').")

df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
df["text"] = df["text"].astype(str).str.strip()

# Map labels to 0/1
if df["label"].dtype == object:
    mapping = {"positive": 1, "negative": 0}
    df["label"] = df["label"].str.lower().map(mapping)
if set(df["label"].unique()) - {0, 1}:
    raise ValueError("Labels must be 0/1 or 'positive'/'negative'.")

# (Optional) keep a balanced subset for quick lessons (comment out to use full data)
pos = df[df["label"] == 1]
neg = df[df["label"] == 0]
n = min(len(pos), len(neg), 3500)  # up to 7k rows total; tweak as needed
df = pd.concat([pos.sample(n=n, random_state=42), neg.sample(n=n, random_state=42)]) \
       .sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"📊 Using {len(df)} rows (balanced subset).")

# Split (80/10/10) with stratify
train_df, tmp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df  = train_test_split(tmp_df, test_size=0.5, random_state=42, stratify=tmp_df["label"])

# Wrap in HF datasets
raw = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(val_df, preserve_index=False),
    "test": Dataset.from_pandas(test_df, preserve_index=False),
})

# Tokenizer (tiny model for speed; can switch to distilbert-base-uncased)
MODEL_NAME = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize with truncation to 128 tokens; pad dynamically later (faster than max_length padding)
def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

tokenized = raw.map(tok, batched=True, remove_columns=["text"])

# Save splits
tokenized["train"].save_to_disk("train_data")
tokenized["validation"].save_to_disk("val_data")
tokenized["test"].save_to_disk("test_data")
print("✅ Saved: train_data, val_data, test_data")