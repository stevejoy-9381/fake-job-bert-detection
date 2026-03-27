import streamlit as st
import requests

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="AI Fake Job Detector",
    layout="wide"
)

# 👉 Change this when you deploy backend
API_URL = "http://127.0.0.1:8000"

# -----------------------
# TITLE
# -----------------------
st.title("🧠 AI Fake Job Detection System")

# -----------------------
# INPUT SECTION
# -----------------------
st.subheader("📄 Enter Job Description")

text = st.text_area(
    "Paste job description here...",
    height=200
)

# -----------------------
# PREDICTION SECTION
# -----------------------
if st.button("🔍 Predict", use_container_width=True):

    if text.strip() == "":
        st.warning("⚠️ Please enter a job description")
    else:
        try:
            with st.spinner("Analyzing..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    json={"description": text}
                )

                result = response.json()

            col1, col2 = st.columns(2)

            # Prediction result
            with col1:
                if result["label"] == "Fake":
                    st.error("⚠️ Fake Job Posting")
                else:
                    st.success("✅ Real Job Posting")

            # Confidence score
            with col2:
                st.metric("Confidence", f"{result['confidence']:.2f}")

        except Exception as e:
            st.error("❌ Backend not running or connection failed")

# -----------------------
# HISTORY SECTION
# -----------------------
st.subheader("📜 Prediction History")

col1, col2, col3 = st.columns(3)

show_all = col1.button("All")
show_fake = col2.button("Fake Only")
show_real = col3.button("Real Only")

try:
    if show_all or show_fake or show_real:

        response = requests.get(f"{API_URL}/history")
        history = response.json()

        if not history:
            st.info("No history found")
        else:
            for item in history[::-1]:  # latest first

                if show_fake and item["label"] != "Fake":
                    continue
                if show_real and item["label"] != "Real":
                    continue

                with st.expander(
                    f"{item['label']} ({item['confidence']:.2f})"
                ):
                    st.write(item["description"])

except:
    st.warning("⚠️ Unable to fetch history (check backend)")