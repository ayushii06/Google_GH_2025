import { useState } from "react";
import { RxCross2 } from "react-icons/rx";

export default function SymptomInput({ onSymptomsChange }) {
    const [input, setInput] = useState("");
    const [symptoms, setSymptoms] = useState(new Set());
    const [showSuggestions, setShowSuggestions] = useState(false);

    // Predefined list of symptoms
    const predefinedSymptoms = ["Chest Pain", "Cough", "Shortness of Breath", 
        "Bone Pain", "Swelling in Joints", "Back Pain", "Injury-related Pain","Severe Headache", "Dizziness or Loss of Balance", "Memory Loss or Confusion","Numbness or Tingling", "Hearing or Vision Problems","Unexplained Seizures","Severe Abdominal Pain", "Head Injury or Trauma", "Weight Loss","Fever", "Fatigue","Nausea or Vomiting"
    ];
    

    // Filter suggestions based on input
    const filteredSuggestions = input
        ? predefinedSymptoms.filter(symptom => symptom.toLowerCase().includes(input.toLowerCase()) && !symptoms.has(symptom))
        : [];

    const addSymptom = (symptom) => {
        if (!symptoms.has(symptom)) {
            setInput("");
            const updatedSymptoms = new Set(symptoms);
            updatedSymptoms.add(symptom);
            setSymptoms(updatedSymptoms);
            onSymptomsChange(Array.from(updatedSymptoms));
            setShowSuggestions(false);
        }
    };

    const removeSymptom = (symptom) => {
        const updatedSymptoms = new Set(symptoms);
        updatedSymptoms.delete(symptom);
        setSymptoms(updatedSymptoms);
        onSymptomsChange(Array.from(updatedSymptoms));
    };

    const handleKeyDown = (event) => {
        if (event.key === "Enter" && input.trim()) {
            event.preventDefault();
            addSymptom(input.trim());
        }
    };

    return (
        <div className="w-full relative">
            <label htmlFor="symptoms" className="text-richblack-200">
                Describe Your Symptoms
            </label>
            <div className="border mt-2 p-2 bg-black text-white rounded-md shadow-sm shadow-white flex flex-wrap gap-2 min-h-[40px]">
                {[...symptoms].map((symptom, index) => (
                    <div key={index} className="bg-blue-500 text-white px-2 py-1 rounded-md flex items-center">
                        {symptom}
                        <button className="ml-2 cursor-pointer" onClick={() => removeSymptom(symptom)}><RxCross2/></button>
                    </div>
                ))}
                <input
                    type="text"
                    id="symptoms"
                    className="bg-black text-white outline-none flex-1"
                    placeholder="Type and press Enter..."
                    value={input}
                    onChange={(e) => {
                        setInput(e.target.value);
                        setShowSuggestions(e.target.value.length > 0 && filteredSuggestions.length > 0);
                    }}
                    onKeyDown={handleKeyDown}
                    onBlur={() => setTimeout(() => setShowSuggestions(false), 200)} 
                />
            </div>

            {/* Suggestions Dropdown */}
            {showSuggestions && (
                <div className="absolute left-0 mt-1 w-full bg-black text-white border border-gray-600 rounded-md shadow-md z-10">
                    {filteredSuggestions.map((symptom, index) => (
                        <div 
                            key={index} 
                            className="p-2 hover:bg-gray-700 cursor-pointer" 
                            onMouseDown={() => addSymptom(symptom)}
                        >
                            {symptom}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
