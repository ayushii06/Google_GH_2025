import os
from models.covid_19.predict_covid19 import predict_covid19
from utils.model_selector import get_relevant_models
from models.brain_tumor.predict_brain_tumor import predict_brain_tumor
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename  # Secure filename handling

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://google-gh-2025-neon.vercel.app"}})

# Directory for temporary image storage
UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Supported model predictions
MODEL_PREDICTORS = {
    "Brain Tumor": predict_brain_tumor,
    "COVID 19": predict_covid19,
}

# route for generating report
@app.route("/generate_report", methods=["POST"])
def generate_report():
    # print("Generating report...", flush=True) 
    
    try:
        # Validate form data
        scan_type = request.form.get("report")
        symptoms = request.form.getlist("symptoms")
        image = request.files.get("files")

        if not scan_type:
            return jsonify({"error": "No report type selected"}), 400

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        if not image:
            return jsonify({"error": "No image file uploaded"}), 400
        
        # Secure filename and save image temporarily
        filename = secure_filename(image.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(img_path)

        print(f"Scan Type: {scan_type}")
        print(f"Symptoms: {symptoms}")
        print(f"Image Saved at: {img_path}")

        # Get relevant models
        relevant_models = get_relevant_models(scan_type, symptoms)

        if not relevant_models:
            return jsonify({"error": f"No AI models available for {scan_type}"}), 400

        # Run predictions
        predictions = []
        for model in relevant_models:
            if model in MODEL_PREDICTORS:
                try:
                    print(f"Running prediction for model: {model}", flush=True)
                    prediction = MODEL_PREDICTORS[model](img_path)
                    confidence = prediction.get("confidence", "N/A") if isinstance(prediction, dict) else "N/A"
                    predictions.append({"model": model, "prediction": prediction, "confidence": confidence})
                except Exception as pred_error:
                    print(f"Error predicting with model {model}: {str(pred_error)}", flush=True)
                    predictions.append({"model": model, "error": f"Failed to predict with {model}"})
            else:
                print(f"Model {model} not found", flush=True)

        # Delete the image after processing to save space
        os.remove(img_path)

        # Response
        return jsonify({
            "status": "success",
            "scan_type": scan_type,
            "symptoms": symptoms,
            "predictions": predictions
        })

    except Exception as e:
        print(f"Error: {str(e)}", flush=True)
        return jsonify({"error": "An unexpected error occurred. Please try again.", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=10000)
