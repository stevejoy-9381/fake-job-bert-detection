import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "stevenson9381/fake-job-bert"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Fake Job Detection")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

text = st.text_area("Enter Job Description")

# -------------------
# Predict Button
# -------------------

if st.button("Predict"):

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()

    label = "Fake" if pred == 1 else "Real"

    # store full description
    st.session_state.history.append({
        "description": text,
        "label": label
    })

    # show only prediction
    if label == "Fake":
        st.error("⚠️ Fake Job Posting")
    else:
        st.success("✅ Real Job Posting")

# -------------------
# History Buttons
# -------------------

st.subheader("History Controls")

col1, col2 = st.columns(2)

show_fake = col1.button("Show Fake History")
show_real = col2.button("Show Real History")

# -------------------
# Fake History
# -------------------

if show_fake:
    st.error("⚠️ Fake Job History")

    for item in st.session_state.history:
        if item["label"] == "Fake":
            st.write(item["description"])
            st.divider()

# -------------------
# Real History
# -------------------

if show_real:
    st.success("✅ Real Job History")

    for item in st.session_state.history:
        if item["label"] == "Real":
            st.write(item["description"])
            st.divider()