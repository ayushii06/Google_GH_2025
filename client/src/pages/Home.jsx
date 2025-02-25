import React from "react"
import IconBtn from "../components/common/IconBtn"
import Modal from "../components/common/Modal"


function page() {
    const [modalVisible, setModalVisible] = React.useState(false);
    
    return(
        <>
        <div className="flex flex-col items-center justify-center h-screen">
            <div className="flex flex-col items-center">
                <div className="text-white text-4xl font-bold mt-4 mb-4"> Beyond Diagnosis, Towards a Healthier You !</div>
                <p className="text-white text-center w-8/12 text-lg mt-2">We know you're not feeling your best—that’s why we’re here. Your health matters, and we’re ready to help. Just share your medical reports or describe your symptoms, and our AI-powered assistant will analyze them instantly.</p>
                <div className="flex gap-4 mt-8">
                    <IconBtn text="Upload your report" onclick={()=>{setModalVisible(true)}}/>
                </div>
            </div>
        </div>

        {modalVisible && (<>
            <div className="fixed top-0 left-0 w-full h-full bg-black bg-opacity-90 flex items-center justify-center">
                <Modal setModalVisible={setModalVisible}/>
            </div>
        </>)}
        </>
    )
}

export default page