import React from "react";
import { useNavigate } from "react-router-dom";
import { useState } from "react";

function Report({ result, image }) {
    console.log("RESULT:", result);
    const [modalVisible, setModalVisible] = useState(false);   

    const handleDownload = () => {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(result, null, 2));
        const element = document.createElement("a");
        element.href = dataStr;
        element.download = "report.json";
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    };

    if (!result || !result.diagnosis) {
        return <div className="text-center text-2xl font-bold my-45 text-gray-500">Our Server is Down. We are sorry !</div>;
    }


    return (
        <div className="my-2 w-8/12 overflow-auto bg-black flex flex-col gap-y-6 rounded-md  px-2 py-1 md:p-8 md:px-12 mx-auto">
            <h2 className="text-2xl font-semibold text-center text-white mb-4">Model Prediction Result</h2>
            <div className="flex justify-center items-center gap-4">
                <div className="space-y-4">
                    {result.diagnosis.map((ele, i) => (
                        <div key={i} className="py-4 px-6 border rounded-lg shadow-md bg-gray-100">
                            <span className="text-xl text-center font-semibold text-blue-600">Prediction {i + 1}</span>
                            <div className="mt-2">
                                <p className="text-gray-700 text-xl">
                                    <strong>Model:</strong> {ele.model}
                                </p>
                                <p className="text-gray-700 text-xl">
                                    <strong>Prediction:</strong> {ele.prediction}
                                </p>
                                {ele?.confidence && ele.confidence>0 && (
                                    <p className="text-gray-700 text-xl">
                                        <strong>Confidence:</strong> {ele.confidence}
                                    </p>
                                )}
                               
                            </div>
                        </div>
                    ))}
                </div>
                
                    <img src={image} className="w-[55%]" alt="" />

            
            </div>
                    <div className="mt-4 text-lg text-center text-white font-medium">
                        <strong>Scan Type:</strong> {result.scan_type}
                    </div>

            
        </div>
    );
}

export default Report;
