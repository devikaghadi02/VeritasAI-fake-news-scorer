# train.py
# This script does everything needed to train the full model:
#   1. Fine-tunes RoBERTa on LIAR statements
#   2. Extracts RoBERTa probability predictions as features
#   3. Combines with stylometric + source features
#   4. Trains a GradientBoosting meta-classifier on top
#   5. Saves everything to the models/ folder

import os
import numpy as np
import pandas as pd
import joblib                                   # for saving sklearn models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import evaluate                                  # HuggingFace evaluate library

from signals import style_features_to_array, get_source_score, ALL_FEATURE_NAMES

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME    = "roberta-base"          # pre-trained model from HuggingFace
NUM_LABELS    = 3                       # REAL, MIXED, FAKE
MAX_LENGTH    = 128                     # max tokens per statement
BATCH_SIZE    = 16                      # how many samples per training step
NUM_EPOCHS    = 3                       # how many full passes over training data
LEARNING_RATE = 2e-5                    # how fast the model learns
OUTPUT_DIR    = "models/roberta"        # where to save the fine-tuned model
META_MODEL_PATH = "models/meta_clf.pkl"

os.makedirs("models", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: CREATE A PYTORCH DATASET CLASS
# HuggingFace Trainer expects data in a specific format.
# We create a class that wraps our DataFrame and tokenizes text on the fly.
# ══════════════════════════════════════════════════════════════════════════════

class LiarDataset(Dataset):
    """
    A PyTorch Dataset that tokenizes LIAR statements for RoBERTa.
    Think of this as a smart list — it gives the model one example at a time.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.statements = df["statement"].tolist()
        self.labels     = df["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Required: tells PyTorch how many samples are in this dataset
        return len(self.statements)

    def __getitem__(self, idx):
        # Required: returns one tokenized sample at index idx
        encoding = self.tokenizer(
            self.statements[idx],
            max_length=self.max_length,
            padding="max_length",       # pad short texts to max_length
            truncation=True,            # cut long texts at max_length
            return_tensors="pt",        # return PyTorch tensors
        )
        return {
            # squeeze() removes the extra dimension added by return_tensors="pt"
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: DEFINE EVALUATION METRICS
# We want to track accuracy and macro-F1 (F1 treats all classes equally).
# Macro-F1 is better than accuracy when classes are imbalanced.
# ══════════════════════════════════════════════════════════════════════════════

accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Called by HuggingFace Trainer after each validation epoch."""
    logits, labels = eval_pred
    # logits are raw scores — argmax picks the highest-scoring class
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1  = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: FINE-TUNE ROBERTA
# ══════════════════════════════════════════════════════════════════════════════

def train_roberta(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Fine-tunes RoBERTa on the LIAR training set.
    Returns the trained model and tokenizer.
    """
    print("\n" + "="*60)
    print("STEP 3: Fine-tuning RoBERTa...")
    print("="*60)

    # Load tokenizer — converts text to token IDs that RoBERTa understands
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Load pre-trained RoBERTa with a classification head on top
    # num_labels=3 adds a layer with 3 outputs (one per class)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    # Wrap our DataFrames in the Dataset class we defined above
    train_dataset = LiarDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset   = LiarDataset(val_df,   tokenizer, MAX_LENGTH)

    # Configure the training process
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,              # regularization to prevent overfitting
        eval_strategy="epoch",    # evaluate after every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,    # keep the best checkpoint
        metric_for_best_model="f1",     # use F1 to decide "best"
        logging_steps=50,               # print loss every 50 steps
        report_to="none",               # don't send logs to wandb/mlflow
    )

    # Trainer handles the training loop for us
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save the final model and tokenizer to disk
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nRoBERTa saved to {OUTPUT_DIR}/")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: EXTRACT ROBERTA PROBABILITY FEATURES
# After fine-tuning, we run RoBERTa on all splits to get probability vectors.
# These become features for the meta-classifier (Stage B).
# ══════════════════════════════════════════════════════════════════════════════

def get_roberta_probs(model, tokenizer, texts: list, batch_size=32) -> np.ndarray:
    """
    Run RoBERTa on a list of texts and return softmax probabilities.

    Returns:
        numpy array of shape (n_samples, 3) — one row per text,
        with [p_real, p_mixed, p_fake] per row.
    """
    model.eval()   # put model in evaluation mode (disables dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_probs = []

    # Process texts in batches to avoid running out of memory
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize the batch
        encodings = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():   # don't compute gradients (saves memory)
            outputs = model(**encodings)

        # outputs.logits shape: (batch_size, 3)
        # Apply softmax to convert logits to probabilities that sum to 1
        probs = torch.softmax(outputs.logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return np.vstack(all_probs)   # stack all batches into one array


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: BUILD THE FULL FEATURE MATRIX
# Combines RoBERTa probs (3) + style features (8) + source score (1) = 12 total
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df: pd.DataFrame, roberta_probs: np.ndarray) -> np.ndarray:
    """
    Assembles the full 12-feature matrix for one DataFrame split.

    Args:
        df: DataFrame with "statement" column (and optionally "url")
        roberta_probs: (n_samples, 3) array of RoBERTa probabilities

    Returns:
        numpy array of shape (n_samples, 12)
    """
    n = len(df)
    style_matrix  = np.zeros((n, 8), dtype=np.float32)
    source_vector = np.zeros((n, 1), dtype=np.float32)

    for i, row in df.iterrows():
        style_matrix[i]  = style_features_to_array(row["statement"])
        url = row.get("url", "")           # some rows may not have a URL
        source_vector[i] = get_source_score(url if isinstance(url, str) else "")

    # Concatenate all three signal arrays horizontally
    # Result shape: (n_samples, 3 + 8 + 1) = (n_samples, 12)
    return np.hstack([roberta_probs, style_matrix, source_vector])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: TRAIN THE META-CLASSIFIER
# Takes the 12-feature vectors and learns the final REAL/MIXED/FAKE decision.
# ══════════════════════════════════════════════════════════════════════════════

def train_meta_classifier(X_train, y_train, X_val, y_val):
    """
    Trains a Gradient Boosting meta-classifier on the fused features.
    """
    print("\n" + "="*60)
    print("STEP 6: Training meta-classifier...")
    print("="*60)

    clf = GradientBoostingClassifier(
        n_estimators=200,       # number of decision trees to build
        max_depth=4,            # how deep each tree can grow
        learning_rate=0.05,     # smaller = more conservative learning
        subsample=0.8,          # use 80% of data per tree (reduces overfitting)
        random_state=42,
        verbose=1,
    )
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    val_preds = clf.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, val_preds, target_names=["REAL", "MIXED", "FAKE"]))

    return clf


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Run everything in order
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    train_df = pd.read_csv("data/train.csv")
    val_df   = pd.read_csv("data/val.csv")
    test_df  = pd.read_csv("data/test.csv")

    # ── Train RoBERTa ─────────────────────────────────────────────────────────
    roberta_model, tokenizer = train_roberta(train_df, val_df)

    # ── Extract RoBERTa probability features for all splits ───────────────────
    print("\nExtracting RoBERTa features...")
    train_probs = get_roberta_probs(roberta_model, tokenizer, train_df["statement"].tolist())
    val_probs   = get_roberta_probs(roberta_model, tokenizer, val_df["statement"].tolist())
    test_probs  = get_roberta_probs(roberta_model, tokenizer, test_df["statement"].tolist())

    # ── Build full feature matrices ───────────────────────────────────────────
    print("\nBuilding feature matrices...")
    X_train = build_feature_matrix(train_df, train_probs)
    X_val   = build_feature_matrix(val_df,   val_probs)
    X_test  = build_feature_matrix(test_df,  test_probs)

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    # ── Train meta-classifier ─────────────────────────────────────────────────
    meta_clf = train_meta_classifier(X_train, y_train, X_val, y_val)

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS")
    print("="*60)
    test_preds = meta_clf.predict(X_test)
    print(classification_report(y_test, test_preds, target_names=["REAL", "MIXED", "FAKE"]))

    # ── Save meta-classifier ──────────────────────────────────────────────────
    joblib.dump(meta_clf, META_MODEL_PATH)
    print(f"\nMeta-classifier saved to {META_MODEL_PATH}")
    print("\nTraining complete!")