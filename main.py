from flask import Flask, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

# Load Hugging Face model
classifier = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save the file temporarily
    file_path = f"C:\Users\Lenovo\Downloads\sad.wav"
    file.save(file_path)

    # Run inference
    results = classifier(file_path)

    # Return the highest scoring emotion
    prediction = max(results, key=lambda x: x["score"])

    return jsonify({"prediction": prediction})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Audio Emotion Recognition API"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
