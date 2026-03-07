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

st.title("Fake Job Detection Dashboard")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

text = st.text_area("Enter Job Description")

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

    st.session_state.history.append({
        "Job Description": text,
        "Prediction": label
    })

# ----------------------
# Dashboard Metrics
# ----------------------

fake_count = sum(1 for x in st.session_state.history if x["Prediction"] == "Fake")
real_count = sum(1 for x in st.session_state.history if x["Prediction"] == "Real")

st.subheader("Dashboard")

col1, col2 = st.columns(2)

col1.metric("⚠️ Fake Jobs Detected", fake_count)
col2.metric("✅ Real Jobs Detected", real_count)

# ----------------------
# Prediction Boxes
# ----------------------

st.subheader("Prediction Results")

fake_col, real_col = st.columns(2)

with fake_col:
    st.error("⚠️ Fake Job Postings")
    for item in st.session_state.history:
        if item["Prediction"] == "Fake":
            st.write(item["Job Description"])

with real_col:
    st.success("✅ Real Job Postings")
    for item in st.session_state.history:
        if item["Prediction"] == "Real":
            st.write(item["Job Description"])

# ----------------------
# History Table
# ----------------------

st.subheader("Prediction History")

df = pd.DataFrame(st.session_state.history)

if not df.empty:
    st.dataframe(df)

# ----------------------
# Download CSV
# ----------------------

if not df.empty:
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name="job_predictions_history.csv",
        mime="text/csv",
    )