from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_PATH = "./model"   # 👉 change this if your model folder name is different

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )
    return classifier