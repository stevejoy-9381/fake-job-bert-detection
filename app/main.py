from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict
from app.db import collection

app = FastAPI()

# Request schema
class JobRequest(BaseModel):
    description: str

# Home route (optional but useful)
@app.get("/")
def home():
    return {"message": "API is running"}

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