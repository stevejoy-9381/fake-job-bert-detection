import gradio as gr
from app.model import predict

def detect(text):
    result = predict(text)
    return f"{result['label']} (Confidence: {result['confidence']:.2f})"

gr.Interface(
    fn=detect,
    inputs=gr.Textbox(lines=6, placeholder="Enter job description"),
    outputs="text",
    title="Fake Job Detection",
    description="Detect fake job postings using AI"
).launch()