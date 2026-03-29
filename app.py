import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Model name (downloads automatically)
MODEL_NAME = "stevenson9381/fake-job-bert"

# Cache model (VERY IMPORTANT for deployment)
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

# UI
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️")

st.title("🕵️ Fake Job Detection System")
st.write("Enter a job description to check if it is Fake or Real")

classifier = load_model()

user_input = st.text_area("Job Description", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter job description")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(user_input)

        st.write(result)  # debug (remove later)

        label = result[0]["label"]
        score = result[0]["score"]

        # Adjust based on your model labels
        if label in ["LABEL_0", "FAKE"]:
            st.error(f"🚨 Fake Job\nConfidence: {score:.2f}")
        else:
            st.success(f"✅ Real Job\nConfidence: {score:.2f}")