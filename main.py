from fastapi import FastAPI, File, UploadFile
import torch
from transformers import pipeline

app = FastAPI()

# Load Hugging Face model
classifier = pipeline("audio-classification", model="Pragnakalp/audio-emotion")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())

    # Run inference
    results = classifier(file.filename)

    # Return the highest scoring emotion
    return {"prediction": max(results, key=lambda x: x["score"])}

# Root endpoint
@app.get("/")
def home():
    return {"message": "Audio Emotion Recognition API"}
