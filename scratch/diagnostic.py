
import sys
import os

print("Starting diagnostic...")

try:
    import streamlit as st
    print(f"Streamlit version: {st.__version__}")
except ImportError:
    print("Streamlit not found")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("Torch not found")

try:
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    print("Transformers imported successfully")
except ImportError:
    print("Transformers not found")

try:
    import joblib
    print("Joblib imported successfully")
except ImportError:
    print("Joblib not found")

from predict import load_roberta, load_meta_clf
print("Predict functions imported")

# Try loading the meta classifier (it's small)
try:
    meta_clf = load_meta_clf()
    print("Meta classifier loaded successfully")
except Exception as e:
    print(f"Error loading meta classifier: {e}")

print("Diagnostic complete.")
