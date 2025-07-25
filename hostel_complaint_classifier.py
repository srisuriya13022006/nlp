
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import requests
# from dotenv import load_dotenv

# # Load .env values (for local dev)
# load_dotenv()

# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend calls

# # HuggingFace API Config
# API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
# HF_TOKEN = os.getenv("HF_TOKEN")

# headers = {
#     "Authorization": f"Bearer {HF_TOKEN}"
# }

# # Complaint labels (tags)
# candidate_labels = [
#     "cleaning", "maintenance", "food", "noise", "staff", "water", "electricity",
#     "wifi issue (slow internet, no connection)", "room condition", "washroom issue",
#     "laundry", "security", "pest control", "air conditioning", "fan or light not working",
#     "bed or furniture damage", "mess timing", "power backup", "waste disposal",
#     "drinking water", "plumbing (tap, pipe leakage)", "other"
# ]

# @app.route("/classify", methods=["POST"])
# def classify():
#     data = request.get_json()
#     complaint_text = data.get("complaint", "").strip()

#     if not complaint_text:
#         return jsonify({"error": "Complaint text is empty"}), 400

#     try:
#         payload = {
#             "inputs": complaint_text,
#             "parameters": {
#                 "candidate_labels": candidate_labels
#             }
#         }

#         response = requests.post(API_URL, headers=headers, json=payload)
#         result = response.json()

#         if "error" in result:
#             return jsonify({"error": result["error"]}), 500

#         predicted_label = result["labels"][0]
#         confidence = round(result["scores"][0], 2)

#         return jsonify({
#             "label": predicted_label,
#             "confidence": confidence
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
import json

# Load .env values (locally)
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS

# HuggingFace API Config
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_TOKEN = os.getenv("HF_TOKEN")

# Gemini API Config
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

hf_headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Complaint Labels
candidate_labels = [
    "cleaning", "maintenance", "food", "noise", "staff", "water", "electricity",
    "wifi issue (slow internet, no connection)", "room condition", "washroom issue",
    "laundry", "security", "pest control", "air conditioning", "fan or light not working",
    "bed or furniture damage", "mess timing", "power backup", "waste disposal",
    "drinking water", "plumbing (tap, pipe leakage)", "other"
]

def query_gemini(complaint_text, hf_label, candidate_labels):
    if not GEMINI_API_KEY:
        return None, "Gemini API key is missing"

    prompt = f"""
    You are a complaint classification assistant. Given the complaint, label from another model, and candidate labels, verify or revise the label.

    Complaint: {complaint_text}
    HF Label: {hf_label}
    Candidates: {', '.join(candidate_labels)}

    Respond in JSON:
    {{
        "label": "<most suitable label>",
        "is_hf_label_correct": <true/false>,
        "explanation": "<short reason>"
    }}
    """

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return None, f"Gemini API error: {response.text}"

    result = response.json()
    text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

    # Clean code block markers
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        gemini_result = json.loads(text)
        if not all(k in gemini_result for k in ["label", "is_hf_label_correct", "explanation"]):
            return None, "Incomplete Gemini response"
        if gemini_result["label"] not in candidate_labels:
            return None, f"Invalid label: {gemini_result['label']}"
        return gemini_result, None
    except json.JSONDecodeError:
        return None, "Invalid JSON from Gemini"

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    complaint_text = data.get("complaint", "").strip()
    if not complaint_text:
        return jsonify({"error": "Complaint text is empty"}), 400

    # HuggingFace Classification
    hf_payload = {
        "inputs": complaint_text,
        "parameters": {
            "candidate_labels": candidate_labels
        }
    }
    hf_response = requests.post(HF_API_URL, headers=hf_headers, json=hf_payload)
    hf_result = hf_response.json()

    if "error" in hf_result:
        return jsonify({"error": hf_result["error"]}), 500

    hf_label = hf_result["labels"][0]
    hf_confidence = round(hf_result["scores"][0], 2)

    # Gemini Cross-check (optional)
    gemini_result, gemini_error = query_gemini(complaint_text, hf_label, candidate_labels)
    if gemini_error:
        return jsonify({
            "label": hf_label,
            "confidence": hf_confidence,
            "warning": f"Gemini check failed: {gemini_error}"
        })

    return jsonify({
        "label": gemini_result["label"],
        "confidence": hf_confidence,
        "is_hf_label_correct": gemini_result["is_hf_label_correct"],
        "explanation": gemini_result["explanation"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
