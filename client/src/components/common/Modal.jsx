import React from "react";
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router-dom";
import IconBtn from "./IconBtn";
import { generateReport } from "../../services/operations/reportAPI";
import SymptomInput from "./SymptomInput";
import LoadingScreen from "../../pages/Loading";
import { useState } from "react";
import Error from "../../pages/Error";
import Report from "./Report";


const report = ['Chest X-Ray', 'MRI','Ultrasound', 'Mammography', 'Cardiac MRI', 'Cardiac CT']

export default function Modal({ setModalVisible }) {
    const [loading, setLoading] = useState(false);
    const [success, setSuccess] = useState(false);
    const [error, setError] = useState(false);
    const [errorMessage, setErrorMessage] = useState("");
    const [result_gen, setResultGen] = useState({});
    const [image, setImage] = useState(null);

    const {
        register,
        handleSubmit,
        setValue,
        formState: { errors },
    } = useForm();

    const submitProfileForm = async (data) => {
        try {
            console.log("DATA - ", data);
    
            // Convert form data into FormData
            const formData = new FormData();
            formData.append("files", data.files[0]); // Get the first file from FileList
            formData.append("report", data.report);
            formData.append("symptoms", JSON.stringify(data.symptoms || [])); // Convert symptoms to JSON
            const imageurl = URL.createObjectURL(data.files[0]);
            setImage(imageurl);

            const result = await generateReport(formData,setLoading,setError,setSuccess,setErrorMessage);
            
            console.log("REPORT - ", result);
            setResultGen(result);
        } catch (error) {
            console.log("ERROR MESSAGE - ", error.message);
        }
    };
    

    return (
<>
        {!loading && !success && !error && (
            <form onSubmit={handleSubmit(submitProfileForm)} encType="multipart/form-data" className="w-6/12 mx-auto">
            <div className="my-2  overflow-auto h-full bg-black  flex flex-col gap-y-6 rounded-md border border-richblack-700 bg-richblack-800 px-2 py-2 md:p-8 md:px-12 mx-auto">
                <h2 className="text-xl  text-center font-semibold text-richblack-5">
                    Fill the following details
                </h2>
                <div className="flex flex-col gap-5 lg:flex-row">
                    <div className="flex flex-col w-full gap-2 ">
                        <label htmlFor="report" className="text-richblack-200">
                            Select Your Medical Report
                        </label>
                        <select
                            name="report"
                            id="report"
                            className="form-style bg-black text-white p-2 rounded-md shadow-sm shadow-white"
                            {...register("report", { required: true })}
                            defaultValue={'Chest X-Ray'}
                        >
                            {report.map((ele, i) => (
                                <option key={i} value={ele}>
                                    {ele}
                                </option>
                            ))}
                        </select>
                        {errors.report && (
                            <span className="-mt-1 text-[12px] text-red-500">
                                Please select the type of your report
                            </span>
                        )}
                    </div>

                </div>

                <div className="flex flex-col gap-5 lg:flex-row">
                    <div className="flex flex-col w-full gap-2 ">
                        <label htmlFor="files" className="text-richblack-200">
                            Upload Your Medical Report
                        </label>
                        {/* files input */}
                        <input
                            type="file"
                            accept="image/*"
                            className="block w-full text-sm text-white file:mr-4 file:py-2 file:px-4 form-style bg-black  shadow-sm shadow-white
                            file:border-0
                            file:text-sm file:font-semibold
                            file:bg-blue-500 file:text-white
                            hover:file:bg-blue-600"
                                           
                            {...register("files", { required: true })}
                        />
                        {errors.files && (
                            <span className="-mt-1 text-[12px] text-red-500">
                                Please upload your report
                            </span>
                        )}
                    </div>

                </div>


                <SymptomInput onSymptomsChange={(symptoms) => setValue("symptoms", symptoms)} />
                {errors.symptoms && (
                    <span className="-mt-1 text-[12px] text-red-500">
                        Please describe your symptoms
                    </span>
                )}


                <div className="flex justify-center gap-2 mt-4">
                    <button
                        type="button"
                        onClick={() => {
                            setModalVisible(false);
                        }}
                        className="cursor-pointer rounded-md border hover:bg-white hover:text-black transition py-2 px-5 font-semibold text-richblack-50"
                    >
                        Cancel
                    </button>
                    <IconBtn type="submit" text="Generate Result" />
                </div>
            </div>
        </form>
        )}

        {loading && <LoadingScreen/> }
        

        {error && <Error message={errorMessage} />}

        {success && (<><Report result={result_gen} image={image}/></>)}
        </>
    );
}
