from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

# 1. SETUP PATHS
# Replace 'vleramm/sentiment-model' with your actual Hugging Face username and repo name
REPO_ID = "vleramm/sentiment-model" 
LOCAL_DIR = "./sentiment-model"

device = 0 if torch.cuda.is_available() else -1

# 2. SMART LOADING LOGIC
try:
    if os.path.exists(LOCAL_DIR):
        print(f"📂 Loading from local folder: {LOCAL_DIR}")
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR)
    else:
        print(f"🌐 Local folder not found. Downloading from Hugging Face: {REPO_ID}")
        model = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
        tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    # 3. ENSURE LABELS ARE PRESENT
    model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # 4. RUN PIPELINE
    clf = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    print("\n🚀 Testing Model:")
    print(f"Review 1: 'I absolutely loved this movie!' -> {clf('I absolutely loved this movie!')}")
    print(f"Review 2: 'The plot was terrible and boring.' -> {clf('The plot was terrible and boring.')}")

except Exception as e:
    print(f"❌ ERROR: Could not load the model. Make sure REPO_ID '{REPO_ID}' is correct.")
    print(f"Details: {e}")