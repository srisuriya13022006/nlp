
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv

# Load .env values (for local dev)
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend calls

# HuggingFace API Config
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Complaint labels (tags)
candidate_labels = [
    "cleaning", "maintenance", "food", "noise", "staff", "water", "electricity",
    "wifi issue (slow internet, no connection)", "room condition", "washroom issue",
    "laundry", "security", "pest control", "air conditioning", "fan or light not working",
    "bed or furniture damage", "mess timing", "power backup", "waste disposal",
    "drinking water", "plumbing (tap, pipe leakage)", "other"
]

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    complaint_text = data.get("complaint", "").strip()

    if not complaint_text:
        return jsonify({"error": "Complaint text is empty"}), 400

    try:
        payload = {
            "inputs": complaint_text,
            "parameters": {
                "candidate_labels": candidate_labels
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        predicted_label = result["labels"][0]
        confidence = round(result["scores"][0], 2)

        return jsonify({
            "label": predicted_label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
