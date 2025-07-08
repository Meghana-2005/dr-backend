from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import urllib.request

app = Flask(__name__)
CORS(app)

# Model download details
MODEL_URL = "https://drive.google.com/uc?export=download&id=1gISF_QUZHIOyvOurwboXi21nuGIOMqHf"
MODEL_PATH = "highest93ri.h5"

# Download the model if it's not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

# Load the model
model = load_model(MODEL_PATH)

# Define class labels
class_labels = [
    "No DR",          # 0
    "Mild",           # 1
    "Moderate",       # 2
    "Severe",         # 3
    "Proliferative DR"  # 4
]

# Preprocessing function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((299, 299))  # Model expects 299x299
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image_bytes = file.read()
    processed_img = preprocess_image(image_bytes)

    prediction = model.predict(processed_img)[0]
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        'predicted_class': predicted_class,
        'predicted_label': class_labels[predicted_class],
        'confidence': round(confidence, 3)
    })

if __name__ == '__main__':
    app.run(debug=True)