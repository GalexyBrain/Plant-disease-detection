from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

# Class Labels
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
               'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
               'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# Load CNN only when needed
cnn_model = None
def get_cnn_model():
    global cnn_model
    if cnn_model is None:
        tf.config.set_visible_devices([], 'GPU')  # Force CNN to run on CPU
        cnn_model = tf.keras.models.load_model("CNN_Model.keras")
    return cnn_model

# Load ViT only when needed
vit_model = None
feature_extractor = None
def get_vit_model():
    global vit_model, feature_extractor
    if vit_model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vit_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=15,
            ignore_mismatched_sizes=True
        )
        vit_model.load_state_dict(torch.load("final_vit_model.pt", map_location=device))
        vit_model.to(device)
        vit_model.eval()
        feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    return vit_model, feature_extractor

# Preprocessing function
def preprocess_image(image_path, model_type):
    image = Image.open(image_path).convert("RGB")

    if model_type == "cnn":
        image = image.resize((96, 103))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    else:
        vit_model, feature_extractor = get_vit_model()
        device = vit_model.device
        image = feature_extractor(images=image, return_tensors="pt")["pixel_values"].to(device)
        return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    model_type = request.form.get("model", "cnn")
    image_path = "temp.jpg"
    file.save(image_path)

    image = preprocess_image(image_path, model_type)

    if model_type == "cnn":
        model = get_cnn_model()
        prediction = model.predict(image)
        if prediction is None or len(prediction) == 0:
            return jsonify({"error": "CNN prediction failed"}), 500
    else:
        model, _ = get_vit_model()
        with torch.no_grad():
            outputs = model(image)
            prediction = outputs.logits.cpu().numpy()
        if prediction is None or prediction.size == 0:
            return jsonify({"error": "ViT prediction failed"}), 500

    predicted_class = class_names[np.argmax(prediction)]
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
