 from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "stevenson9381/fake-job-bert"


# ✅ Lazy loading (prevents memory crash at startup)
@lru_cache()
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()  # important for inference
    return model, tokenizer


def predict(text: str):
    model, tokenizer = get_model()

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

    pred = torch.argmax(probs, dim=1).item()

    label = "Fake" if pred == 1 else "Real"
    confidence = probs[0][pred].item()

    return {
        "label": label,
        "confidence": float(confidence)
    }