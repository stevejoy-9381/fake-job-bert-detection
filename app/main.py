import os
from fastapi import FastAPI
from pydantic import BaseModel
 

from app.model import predict
from app.db import collection

app = FastAPI()

# Optional: Hugging Face / environment port (not required with Docker CMD)
port = int(os.environ.get("PORT", 7860))

# Request schema
class JobRequest(BaseModel):
    description: str

# Home route
@app.get("/")
def home():
    return {"message": "Fake Job Detection API is running"}

# Predict + Store
@app.post("/predict")
def predict_job(request: JobRequest):
    result = predict(request.description)

    data = {
        "description": request.description,
        "label": result["label"],
        "confidence": result["confidence"]
    }

    collection.insert_one(data)

    return result

# Get History
@app.get("/history")
def get_history():
    data = list(collection.find({}, {"_id": 0}))
    return data