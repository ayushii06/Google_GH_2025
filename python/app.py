import os
from models.covid_19.predict_covid19 import predict_covid19
from utils.model_selector import get_relevant_models
from models.brain_tumor.predict_brain_tumor import predict_brain_tumor
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename  # Secure filename handling


app = Flask(__name__)
CORS(app, resources={r"/*": { "origins": "https://google-gh-2025-neon.vercel.app", "methods": ["GET", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"], "allow_headers": [ "X-Requested-With", "Content-Type", "Access-Control-Allow-Headers", "Access-Control-Allow-Origin", "Access-Control-Allow-Methods", "Access-Control-Allow-Credentials", "Authorization" ], "supports_credentials": True }})  

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PREDICTORS = {
    "Brain Tumor": predict_brain_tumor,
    "COVID 19": predict_covid19,
}

# ✅ Handle Preflight Requests (Important for FormData)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route("/generate_report", methods=["OPTIONS", "POST"])
def generate_report():
    if request.method == "OPTIONS":
        # ✅ Handle CORS Preflight Request
        response = jsonify({"message": "CORS preflight passed"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response, 200

    try:
        scan_type = request.form.get("report")
        symptoms = request.form.getlist("symptoms")
        image = request.files.get("files")

        if not scan_type:
            return jsonify({"error": "No report type selected"}), 400
        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400
        if not image:
            return jsonify({"error": "No image file uploaded"}), 400

        filename = secure_filename(image.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(img_path)

        relevant_models = get_relevant_models(scan_type, symptoms)
        if not relevant_models:
            return jsonify({"error": f"No AI models available for {scan_type}"}), 400

        predictions = []
        for model in relevant_models:
            if model in MODEL_PREDICTORS:
                try:
                    prediction = MODEL_PREDICTORS[model](img_path)
                    confidence = prediction.get("confidence", "N/A") if isinstance(prediction, dict) else "N/A"
                    predictions.append({"model": model, "prediction": prediction, "confidence": confidence})
                except Exception as pred_error:
                    predictions.append({"model": model, "error": f"Failed to predict with {model}"})
            else:
                print(f"Model {model} not found", flush=True)

        os.remove(img_path)

        return jsonify({
            "status": "success",
            "scan_type": scan_type,
            "symptoms": symptoms,
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)

