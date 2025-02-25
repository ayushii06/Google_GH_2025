def get_relevant_models(scan_type, symptoms):
    print(f"[INFO] Finding relevant models for scan type: {scan_type}, Symptoms: {symptoms}")

    # Normalize inputs to lowercase
    scan_type = scan_type.lower()
    symptoms = [symptom.lower() for symptom in symptoms]  # Convert symptoms to lowercase

    # Define which models support which scan types
    SCAN_MODEL_MAPPING = {
        "mri": ["Brain Tumor"],
        "chest x-ray": ["COVID 19", "Pneumonia"],
        "ct scan": ["Lung Cancer"]
    }

    # Define symptoms related to each model
    MODEL_SYMPTOM_MAPPING = {
        "Brain Tumor": ["headache", "seizures", "nausea", "vision problems"],
        "COVID 19": ["cough", "fever", "breathing issues", "fatigue"],
        "Pneumonia": ["cough", "chest pain", "shortness of breath", "fatigue"],
        "Lung Cancer": ["persistent cough", "weight loss", "chest pain"]
    }

    # Get models allowed for the given scan type
    allowed_models = SCAN_MODEL_MAPPING.get(scan_type, [])
    print(f"[INFO] Allowed models for scan type '{scan_type}': {allowed_models}")

    # Score models based on symptom matches
    symptom_counts = {}
    for model in allowed_models:
        match_count = len(set(symptoms) & set(MODEL_SYMPTOM_MAPPING.get(model, [])))
        symptom_counts[model] = match_count  # Even if match_count = 0, store it

    # Sort models by highest symptom match count
    sorted_models = sorted(symptom_counts.keys(), key=lambda m: symptom_counts[m], reverse=True)

    # If no symptoms match, return default models for the scan type
    if not sorted_models or all(count == 0 for count in symptom_counts.values()):
        print(f"[WARNING] No symptoms matched. Returning default models: {allowed_models}")
        return allowed_models

    return sorted_models
