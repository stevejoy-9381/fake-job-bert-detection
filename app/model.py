from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "stevenson9381/fake-job-bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def predict(text: str):
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
    confidence = probs[0][pred].item()

    return {
        "label": label,
        "confidence": confidence
    }