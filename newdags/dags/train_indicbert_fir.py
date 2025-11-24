import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# =====================
# CONFIG
# =====================
# Path to your cleaned FIR data
DATA_PATH = os.getenv("FIR_DATA_PATH", "crimeData_cleaned.csv")

# Max number of rows to use for training (randomly sampled)
MAX_SAMPLES = 50_000

# IndicBERT base model name from HuggingFace
MODEL_NAME = "ai4bharat/indic-bert"  # :contentReference[oaicite:1]{index=1}

# Where to save the fine-tuned model
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./indicbert_fir_model")

# Sequence length and batch sizes tuned for low RAM
MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 2  # start small; you can increase later


# =====================
# DATASET
# =====================
class FIRDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        brief = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            brief,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# =====================
# METRICS
# =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


# =====================
# MAIN TRAINING FUNCTION
# =====================
def train_indicbert_fir():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found at {DATA_PATH}")

    print(f"📄 Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required_cols = {"FIR_BRIEF", "ACT", "SEC_OF_LAW"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Create target label string: ACT||SEC_OF_LAW
    df["Target"] = df["ACT"].astype(str) + "||" + df["SEC_OF_LAW"].astype(str)

    # Drop rows with empty brief or target
    df = df.dropna(subset=["FIR_BRIEF", "Target"])
    df = df[df["FIR_BRIEF"].astype(str).str.strip() != ""]
    df = df[df["Target"].astype(str).str.strip() != ""]

    print(f"✅ After cleaning, rows: {len(df)}")

    # Randomly sample up to MAX_SAMPLES rows
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"🎯 Randomly sampled {MAX_SAMPLES} rows for training.")
    else:
        print(f"🎯 Using all {len(df)} rows (less than {MAX_SAMPLES}).")

    # Encode labels
    label_strings = df["Target"].unique().tolist()
    label2id: Dict[str, int] = {lbl: i for i, lbl in enumerate(sorted(label_strings))}
    id2label: Dict[int, str] = {i: lbl for lbl, i in label2id.items()}
    num_labels = len(label2id)

    print(f"🔢 Number of distinct ACT||SEC labels: {num_labels}")

    df["label_id"] = df["Target"].map(label2id)

    # Train/test split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"] if num_labels > 1 else None,
    )

    print(f"📊 Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    # Load tokenizer & model
    print(f"🧠 Loading IndicBERT model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Build datasets
    train_dataset = FIRDataset(
        texts=train_df["FIR_BRIEF"].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
    )

    val_dataset = FIRDataset(
        texts=val_df["FIR_BRIEF"].tolist(),
        labels=val_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
    )

    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training arguments (CPU-friendly)
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        remove_unused_columns=True,
        fp16=False,          # keep off on CPU
        no_cuda=True,        # force CPU (Docker on Mac, low RAM)
        report_to=[],        # no wandb/mlflow integration here
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print("🚀 Starting IndicBERT fine-tuning...")
    train_result = trainer.train()
    print("✅ Training complete.")

    metrics = trainer.evaluate()
    print("📈 Validation metrics:", metrics)

    # Save final model + tokenizer for deployment
    print(f"💾 Saving model & tokenizer to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save label mappings so predictor can decode ACT & section
    label_map_path = os.path.join(OUTPUT_DIR, "label_mapping.txt")
    with open(label_map_path, "w", encoding="utf-8") as f:
        for idx, label_str in id2label.items():
            f.write(f"{idx}\t{label_str}\n")
    print(f"💾 Saved label mapping to {label_map_path}")

    print("🎉 Done. You can now use this model folder in your predictor service.")


if __name__ == "__main__":
    train_indicbert_fir()
