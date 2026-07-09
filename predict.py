# predict.py
# This file loads the saved models and exposes a single predict() function
# that the Streamlit app calls. It also handles URL article extraction.

import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification
from newspaper import Article                    # extracts article from URL
from signals import (
    style_features_to_array,
    get_source_score,
)

# ── Configuration (must match train.py) ───────────────────────────────────────
ROBERTA_DIR = "Devika2006/fake-news-credibility-roberta"
META_MODEL_PATH = "models/meta_clf.pkl"
MAX_LENGTH      = 128
LABEL_NAMES     = ["REAL", "MIXED", "FAKE"]

# Map label index to human-readable verdict and color
LABEL_INFO = {
    0: {"name": "LIKELY REAL",  "color": "#2ecc71", "emoji": "✅"},
    1: {"name": "MIXED",        "color": "#f39c12", "emoji": "⚠️"},
    2: {"name": "LIKELY FAKE",  "color": "#e74c3c", "emoji": "🚨"},
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# We use @st.cache_resource in the app to load these only once.
# Here we provide plain load functions.
# ══════════════════════════════════════════════════════════════════════════════

def load_roberta():
    """Load the fine-tuned RoBERTa model and tokenizer from disk."""
    print("Loading RoBERTa model...")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_DIR)
    model = RobertaForSequenceClassification.from_pretrained(ROBERTA_DIR)
    model.eval()   # evaluation mode — disables dropout
    return model, tokenizer


def load_meta_clf():
    """Load the trained GradientBoosting meta-classifier."""
    print("Loading meta-classifier...")
    return joblib.load(META_MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# ARTICLE EXTRACTION FROM URL
# Uses the newspaper3k library to pull title + body text from any news URL.
# ══════════════════════════════════════════════════════════════════════════════

def extract_article(url: str) -> dict:
    """
    Download and parse a news article from a URL.

    Args:
        url: Full URL of a news article.

    Returns:
        dict with keys: title, text, source_url, success (bool), error (str)
    """
    try:
        article = Article(url)
        article.download()     # fetch the HTML from the web
        article.parse()        # extract title, body, images, etc.

        if not article.text:
            return {"success": False, "error": "Could not extract article text."}

        return {
            "success":    True,
            "title":      article.title or "",
            "text":       article.text,
            "source_url": url,
            "error":      None,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "title": "", "text": "", "source_url": url}


# ══════════════════════════════════════════════════════════════════════════════
# CORE PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(
    text: str,
    url: str,
    roberta_model,
    tokenizer,
    meta_clf,
) -> dict:
    """
    Run the full 3-signal pipeline and return credibility scores.

    Args:
        text:          The article body or statement text to analyze.
        url:           The source URL (can be empty string).
        roberta_model: Loaded RoBERTa model.
        tokenizer:     Loaded RoBERTa tokenizer.
        meta_clf:      Loaded meta-classifier.

    Returns:
        dict with all scores, predictions, and feature values for the app.
    """

    # ── Signal 1: RoBERTa probability vector ─────────────────────────────────
    # Tokenize the input text
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",   # PyTorch tensors
    )

    device = next(roberta_model.parameters()).device
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = roberta_model(**encoding)

    # Convert raw logits to probabilities (3 values that sum to 1.0)
    roberta_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    # roberta_probs = [p_real, p_mixed, p_fake]

    # ── Signal 2: Stylometric features ───────────────────────────────────────
    style_feats = style_features_to_array(text)   # shape: (8,)

    # ── Signal 3: Source reputation ───────────────────────────────────────────
    source_score = get_source_score(url)           # single float 0.0–1.0

    # ── Assemble the 12-feature vector ────────────────────────────────────────
    feature_vector = np.hstack([
        roberta_probs,                # 3 features
        style_feats,                  # 8 features
        np.array([source_score]),     # 1 feature
    ]).reshape(1, -1)                 # shape (1, 12) for sklearn

    # ── Meta-classifier prediction ────────────────────────────────────────────
    # predict_proba returns probability for each class: [p_real, p_mixed, p_fake]
    meta_probs   = meta_clf.predict_proba(feature_vector)[0]
    predicted_label = int(np.argmax(meta_probs))

    # ── Credibility score (0–100 scale) ───────────────────────────────────────
    # We use the probability of being REAL as the credibility score.
    # Higher = more credible.
    credibility_score = round(float(meta_probs[0]) * 100, 1)

    # ── Build signal contributions for the app display ────────────────────────
    # How confident was each signal individually?
    signal_breakdown = {
        "RoBERTa (text analysis)": {
            "real":  round(float(roberta_probs[0]) * 100, 1),
            "mixed": round(float(roberta_probs[1]) * 100, 1),
            "fake":  round(float(roberta_probs[2]) * 100, 1),
        },
        "Writing style": {
            # Style doesn't directly give probabilities — we derive a simple
            # credibility proxy: fewer emotional words + more hedging = more credible
            "style_score": _style_credibility_score(style_feats),
        },
        "Source reputation": {
            "score": round(source_score * 100, 1),
        },
    }

    return {
        "credibility_score":  credibility_score,
        "predicted_label":    predicted_label,
        "label_name":         LABEL_INFO[predicted_label]["name"],
        "label_color":        LABEL_INFO[predicted_label]["color"],
        "label_emoji":        LABEL_INFO[predicted_label]["emoji"],
        "meta_probs":         {
            "real":  round(float(meta_probs[0]) * 100, 1),
            "mixed": round(float(meta_probs[1]) * 100, 1),
            "fake":  round(float(meta_probs[2]) * 100, 1),
        },
        "signal_breakdown":   signal_breakdown,
        "feature_vector":     feature_vector,    # for SHAP in the app
        "roberta_probs":      roberta_probs,
        "style_feats":        style_feats,
        "source_score":       source_score,
    }


def _style_credibility_score(style_feats: np.ndarray) -> float:
    """
    Derive a simple 0–100 credibility score from writing style features.
    High hedging + low emotional words + low caps = more credible.

    style_feats order: exclamation, caps, avg_word_len, question,
                       hedging, emotional, unique_word, sentence_len
    """
    exclamation = style_feats[0]
    caps        = style_feats[1]
    hedging     = style_feats[4]
    emotional   = style_feats[5]

    # Penalize high exclamation and caps; reward hedging; penalize emotional
    raw_score = (
        - 30 * min(exclamation, 0.1) / 0.1     # up to -30 points for exclamations
        - 20 * min(caps, 0.3) / 0.3             # up to -20 points for ALL CAPS
        + 20 * min(hedging, 2.0) / 2.0          # up to +20 points for hedging
        - 20 * min(emotional, 2.0) / 2.0        # up to -20 points for emotional words
        + 50                                     # base score of 50
    )
    return round(max(0.0, min(100.0, raw_score)), 1)