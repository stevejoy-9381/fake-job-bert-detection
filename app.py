import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Model name
MODEL_NAME = "stevenson9381/fake-job-bert"

# Page configuration
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")

st.title("🕵️ Fake Job Detection System")
st.write("Enter a job description to check if it is Fake or Real")

# Load model (cached)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )
    return classifier

classifier = load_model()

# Input box
user_input = st.text_area("Job Description", height=200)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter job description")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(user_input)

        label = result[0]["label"]
        score = result[0]["score"]

        # Adjust label mapping if needed
        if label in ["LABEL_1", "FAKE"]:
            st.error(f"🚨 Fake Job (Confidence: {score:.2f})")
        else:
            st.success(f"✅ Real Job (Confidence: {score:.2f})")