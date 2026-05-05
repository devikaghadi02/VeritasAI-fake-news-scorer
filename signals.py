# signals.py
# Extracts the 3 independent credibility signals for each article/statement:
#   Signal 1 → Stylometric (writing style) features
#   Signal 2 → Source reputation score
#   Signal 3 → RoBERTa embedding probabilities (handled in train.py)

import re
import numpy as np
from urllib.parse import urlparse


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 1 — STYLOMETRIC FEATURES
# These capture writing patterns known to differ between real and fake news.
# Fake news tends to use more caps, exclamation marks, emotional words, etc.
# ══════════════════════════════════════════════════════════════════════════════

# Words that hedge claims — real journalism uses these; fake news avoids them
HEDGING_WORDS = [
    "allegedly", "reportedly", "according to", "sources say",
    "claimed", "suggested", "appears to", "seems to", "unconfirmed"
]

# Words that signal emotional manipulation — more common in fake news
EMOTIONAL_WORDS = [
    "shocking", "unbelievable", "bombshell", "breaking", "exposed",
    "secret", "conspiracy", "cover-up", "scandal", "outrage", "hoax",
    "mainstream media", "they don't want you to know", "wake up"
]


def extract_style_features(text: str) -> dict:
    """
    Extract stylometric features from a piece of text.

    Args:
        text: The article body or statement text.

    Returns:
        A dictionary of numeric features. Each value is a float.
    """
    if not text or not text.strip():
        # If text is empty, return zeros for all features
        return {k: 0.0 for k in [
            "exclamation_ratio", "caps_ratio", "avg_word_length",
            "question_ratio", "hedging_score", "emotional_score",
            "unique_word_ratio", "sentence_avg_length"
        ]}

    words = text.split()                    # split into individual words
    sentences = re.split(r'[.!?]+', text)  # split into sentences
    sentences = [s.strip() for s in sentences if s.strip()]  # remove empty

    total_words = max(len(words), 1)        # avoid dividing by zero
    total_chars = max(len(text), 1)
    total_sentences = max(len(sentences), 1)

    text_lower = text.lower()

    # Ratio of exclamation marks to total words
    # High value = sensationalist tone
    exclamation_ratio = text.count("!") / total_words

    # Ratio of UPPERCASE characters to total characters
    # High value = shouting / aggression
    caps_ratio = sum(1 for c in text if c.isupper()) / total_chars

    # Average word length — longer words = more formal/technical writing
    avg_word_length = sum(len(w) for w in words) / total_words

    # Ratio of question marks to total words
    # Fake news uses rhetorical questions to plant doubt
    question_ratio = text.count("?") / total_words

    # How many hedging phrases appear per 100 words
    # Real journalism hedges uncertain claims
    hedging_count = sum(1 for hw in HEDGING_WORDS if hw in text_lower)
    hedging_score = hedging_count / (total_words / 100)

    # How many emotional/manipulative words appear per 100 words
    # Fake news uses emotional triggers to bypass critical thinking
    emotional_count = sum(1 for ew in EMOTIONAL_WORDS if ew in text_lower)
    emotional_score = emotional_count / (total_words / 100)

    # Ratio of unique words to total words (vocabulary richness)
    # Low value = repetitive language (common in propaganda)
    unique_word_ratio = len(set(w.lower() for w in words)) / total_words

    # Average words per sentence
    # Very long sentences = complex writing; very short = punchy/tabloid style
    sentence_avg_length = total_words / total_sentences

    return {
        "exclamation_ratio":   round(exclamation_ratio, 4),
        "caps_ratio":          round(caps_ratio, 4),
        "avg_word_length":     round(avg_word_length, 4),
        "question_ratio":      round(question_ratio, 4),
        "hedging_score":       round(hedging_score, 4),
        "emotional_score":     round(emotional_score, 4),
        "unique_word_ratio":   round(unique_word_ratio, 4),
        "sentence_avg_length": round(sentence_avg_length, 4),
    }


def style_features_to_array(text: str) -> np.ndarray:
    """Convert style features dict to a numpy array for model input."""
    feats = extract_style_features(text)
    # Always return features in the same order
    return np.array([
        feats["exclamation_ratio"],
        feats["caps_ratio"],
        feats["avg_word_length"],
        feats["question_ratio"],
        feats["hedging_score"],
        feats["emotional_score"],
        feats["unique_word_ratio"],
        feats["sentence_avg_length"],
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL 2 — SOURCE REPUTATION SCORE
# A lookup table of known news domains and their credibility scores (0–1).
# Built from publicly available media bias / fact-check databases.
# ══════════════════════════════════════════════════════════════════════════════

# Scores are on a 0.0 (completely unreliable) to 1.0 (highly credible) scale.
# Sources not in this table get a neutral score of 0.5.
SOURCE_REPUTATION = {
    # ── Highly credible sources ──────────────────────────────────────────────
    "reuters.com":          0.97,
    "apnews.com":           0.97,
    "bbc.com":              0.95,
    "bbc.co.uk":            0.95,
    "npr.org":              0.93,
    "theguardian.com":      0.92,
    "nytimes.com":          0.91,
    "washingtonpost.com":   0.91,
    "economist.com":        0.93,
    "ft.com":               0.93,
    "wsj.com":              0.90,
    "bloomberg.com":        0.91,
    "nature.com":           0.98,
    "science.org":          0.98,
    "who.int":              0.96,
    "cdc.gov":              0.96,
    "gov.uk":               0.94,
    "usa.gov":              0.94,

    # ── Mixed / partisan but generally factual ───────────────────────────────
    "cnn.com":              0.72,
    "foxnews.com":          0.60,
    "msnbc.com":            0.65,
    "nypost.com":           0.58,
    "dailymail.co.uk":      0.52,
    "huffpost.com":         0.65,
    "vox.com":              0.70,
    "theatlantic.com":      0.78,
    "politico.com":         0.80,
    "vice.com":             0.62,

    # ── Low credibility / known misinformation sources ───────────────────────
    "infowars.com":         0.04,
    "naturalnews.com":      0.06,
    "breitbart.com":        0.18,
    "thegatewaypundit.com": 0.08,
    "worldnewsdailyreport.com": 0.03,
    "empirenews.net":       0.03,
    "nationalreport.net":   0.03,
    "beforeitsnews.com":    0.10,
    "yournewswire.com":     0.05,
    "activistpost.com":     0.15,
    "zerohedge.com":        0.22,
    "rt.com":               0.25,     # state-funded propaganda
    "sputniknews.com":      0.20,
}


def get_source_score(url: str) -> float:
    """
    Look up the credibility score for a given URL's domain.

    Args:
        url: Full URL like "https://www.reuters.com/article/..."

    Returns:
        Float between 0.0 and 1.0. Unknown sources return 0.5 (neutral).
    """
    if not url or not url.strip():
        return 0.5   # no URL provided → neutral

    try:
        # Extract just the domain from the full URL
        # e.g. "https://www.bbc.com/news/..." → "bbc.com"
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        domain = domain.replace("www.", "")    # remove www. prefix

        # Direct lookup first
        if domain in SOURCE_REPUTATION:
            return SOURCE_REPUTATION[domain]

        # Try matching any subdomain: "blogs.reuters.com" → "reuters.com"
        parts = domain.split(".")
        if len(parts) > 2:
            root_domain = ".".join(parts[-2:])   # keep last two parts
            if root_domain in SOURCE_REPUTATION:
                return SOURCE_REPUTATION[root_domain]

        return 0.5   # unknown source → neutral score

    except Exception:
        return 0.5   # if anything goes wrong → neutral


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE NAME REGISTRY
# Used when displaying SHAP/LIME explanations in the app.
# The order here must match the order features are assembled in train.py.
# ══════════════════════════════════════════════════════════════════════════════

# Names for the 3 RoBERTa probability outputs
ROBERTA_FEATURE_NAMES = ["roberta_prob_real", "roberta_prob_mixed", "roberta_prob_fake"]

# Names for the 8 stylometric features (same order as style_features_to_array)
STYLE_FEATURE_NAMES = [
    "exclamation_ratio",
    "caps_ratio",
    "avg_word_length",
    "question_ratio",
    "hedging_score",
    "emotional_score",
    "unique_word_ratio",
    "sentence_avg_length",
]

# Name for the single source score
SOURCE_FEATURE_NAMES = ["source_credibility_score"]

# All feature names in order (total = 3 + 8 + 1 = 12)
ALL_FEATURE_NAMES = ROBERTA_FEATURE_NAMES + STYLE_FEATURE_NAMES + SOURCE_FEATURE_NAMES