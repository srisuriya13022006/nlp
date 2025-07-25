from transformers import pipeline
import torch
import tkinter as tk
from tkinter import messagebox

# Initialize the zero-shot classification pipeline using PyTorch
def load_classifier():
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            framework="pt"
        )
        return classifier
    except Exception as e:
        messagebox.showerror("Model Load Error", f"Error loading model: {e}")
        return None

candidate_labels = [
    "cleaning",
    "maintenance",
    "food",
    "noise",
    "staff",
    "water",
    "electricity",
    "wifi issue (slow internet, no connection)",
    "room condition",
    "washroom issue",
    "laundry",
    "security",
    "pest control",
    "air conditioning",
    "fan or light not working",
    "bed or furniture damage",
    "mess timing",
    "power backup",
    "waste disposal",
    "drinking water",
    "plumbing (tap, pipe leakage)",  
    "other"
]


# Function to classify a complaint
def classify_complaint(complaint_text):
    if not complaint_text.strip():
        messagebox.showwarning("Input Error", "Please enter a complaint.")
        return

    try:
        result = classifier(complaint_text, candidate_labels, multi_label=False)
        predicted_label = result['labels'][0]
        confidence = result['scores'][0]
        output_label.config(text=f"Label: {predicted_label}\nConfidence: {confidence:.2f}")
    except Exception as e:
        messagebox.showerror("Classification Error", f"Error: {e}")

# Load classifier model once globally
classifier = load_classifier()

# GUI
root = tk.Tk()
root.title("Hostel Complaint Classifier")
root.geometry("500x300")
root.resizable(False, False)

tk.Label(root, text="Enter Hostel Complaint:", font=("Arial", 12)).pack(pady=10)

text_entry = tk.Text(root, height=4, width=50, font=("Arial", 11))
text_entry.pack()

tk.Button(root, text="Classify", font=("Arial", 12), bg="#4CAF50", fg="white",
          command=lambda: classify_complaint(text_entry.get("1.0", tk.END))).pack(pady=10)

output_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
output_label.pack(pady=10)

root.mainloop()
