import tensorflow as tf
from .preprocess import preprocess_brain_tumor
import numpy as np
from tensorflow.keras.models import load_model

# Load model globally to avoid reloading every time
model = load_model('models/brain_tumor/brain_tumor_detector.h5')

def predict_brain_tumor(image_path):
    print("Predicting...")

    # Preprocess image
    processed_image = preprocess_brain_tumor(image_path)
    
    if processed_image is None:
        return {"error": "Image preprocessing failed!"}

    print(f"Processed Image Shape: {processed_image.shape}", flush=True)

    # Make prediction
    pred = model.predict(processed_image)[0][0]  # Extract scalar prediction
    print(f"Raw Prediction Score: {pred}", flush=True)
    
    # Map class index to class labels
    class_labels = {0: "no", 1: "yes"}
    
    # Determine class and confidence
    predicted_class = 1 if pred > 0.5 else 0
    confidence = round(float(pred if predicted_class == 1 else 1 - pred), 4)  # Confidence score

    result = {
        "class": class_labels[predicted_class],
        "confidence": confidence*100,
        "prediction": predicted_class,
    }

    return result
