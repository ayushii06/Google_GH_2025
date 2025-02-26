import { toast } from "react-hot-toast"
import { apiConnector } from "../apiconnector"
import { diagnosisEndpoints } from "../api"



const { GENERATE_REPORT_API, } = diagnosisEndpoints;

export const generateReport = async(formData, setLoading, setError, setSuccess, setErrorMessage) => {
        let report = {};
        console.log("GENERATE REPORT API REQUEST1............", formData);
        try {
            if (!formData || !(formData instanceof FormData)) {
                throw new Error("Invalid form data provided.");
            }

            setLoading(true);
            setError(false);
            setSuccess(false);

            console.log("GENERATE REPORT API REQUEST............", formData);

            const response = await apiConnector("POST", GENERATE_REPORT_API, formData);

            console.log("GENERATE REPORT API RESPONSE............", response);

            if (!response || response.status !== 200) {
                throw new Error(response?.data?.message || "Failed to generate report. Please try again.");
            }

            const { predictions, scan_type, scan_image, symptoms } = response.data;

            if (!predictions || !Array.isArray(predictions)) {
                throw new Error("Invalid response format: Predictions data missing.");
            }

            report = {
                diagnosis: predictions.map(pred => ({
                    model: pred.model || "Unknown Model",
                    prediction: pred.prediction || "N/A",
                    confidence: pred.confidence || "N/A",
                })),
                scan_type: scan_type || "Unknown",
                scan_image: scan_image || "N/A",
                symptoms: JSON.stringify(symptoms || []),
            };

            console.log("Formatted REPORT for saving............", report);

            
            
            setSuccess(true);
            toast.success("Report generated successfully");
        } catch (error) {
            console.error("GENERATE REPORT API ERROR............", error);
            setError(true);
            setErrorMessage(error.message || "Unexpected error while generating report");
            toast.error(error.message || "Error while generating report");
        } finally {
            setLoading(false);
        }
        return report;
    };
