import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Page config
st.set_page_config(page_title="Fake Job Detector", page_icon="🔍")

# Title
st.title("🔍 Fake Job Detector")
st.markdown("Paste a job description to check if it's real or fake")

# Load model (cached for performance)
@st.cache_resource
def load_model():
    model_name = "stevenson9381/fake-job-bert"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# Load once
tokenizer, model = load_model()

# Text input
job_text = st.text_area("Job Description", height=200, placeholder="Paste job description here...")

# Analyze button
if st.button("🔍 Analyze", type="primary"):
    if job_text.strip():
        # Tokenize and predict
        inputs = tokenizer(job_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            confidence = torch.softmax(outputs.logits, dim=1)[0][prediction].item()
        
        # Show result
        if prediction == 0:  # Real job
            st.success(f"✅ **REAL JOB**")
            st.write(f"Confidence: {confidence:.1%}")
        else:  # Fake job
            st.error(f"⚠️ **FAKE JOB**")
            st.write(f"Confidence: {confidence:.1%}")
    else:
        st.warning("Please paste a job description")