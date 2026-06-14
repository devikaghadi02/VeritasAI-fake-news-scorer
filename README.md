# VeritasAI — Fake News Credibility Scorer

A multi-signal AI system that analyzes news articles and statements for credibility using three independent signals: deep language understanding, writing style analysis, and source reputation scoring.

## How It Works
The app uses a 3-signal pipeline to score credibility:
1. **RoBERTa text model** — Fine-tuned transformer that reads the actual words and detects linguistic patterns associated with misinformation
2. **Writing style analysis** — Detects sensationalist patterns like excessive caps, emotional language, and exclamation marks
3. **Source reputation** — Cross-references the article domain against a curated credibility database of 50+ news sources

All 3 signals are fused by a Gradient Boosting meta-classifier that outputs a final credibility score from 0–100.

## Tech Stack
- **Model:** RoBERTa (HuggingFace Transformers)
- **Meta-classifier:** Gradient Boosting (scikit-learn)
- **Explainability:** SHAP, LIME
- **App:** Streamlit + Plotly
- **Dataset:** LIAR dataset (12,800 labeled political statements)

## Project Structure
├── app.py           # Streamlit app
├── train.py         # RoBERTa fine-tuning + meta-classifier training
├── predict.py       # Prediction pipeline
├── signals.py       # Feature extraction (style + source signals)
├── data_prep.py     # Data loading and preprocessing
└── requirements.txt

## Run Locally
```bash
git clone https://github.com/devikaghadi02/VeritasAI-fake-news-scorer.git
cd VeritasAI-fake-news-scorer
pip install -r requirements.txt
streamlit run app.py
```

## Model
The fine-tuned RoBERTa model is hosted on HuggingFace:
[Devika2006/fake-news-credibility-roberta](https://huggingface.co/Devika2006/fake-news-credibility-roberta)
