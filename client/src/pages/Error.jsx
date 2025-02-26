import React from 'react';
import { DotLottieReact } from '@lottiefiles/dotlottie-react';


const Error = (message) => {
  return (
    <>
      <div className="bg-black mx-auto my-20 text-white px-10 py-10">
        <DotLottieReact className='w-80 my-10 mx-auto font-bold'
          src="https://lottie.host/cdcd5c2a-b2d8-4b15-8505-beb78e717d6b/jrIik8tJpI.lottie"
          loop
          autoplay
        />
        <div className="text-center mt-5 mb-5 font-medium text-2xl ">
          <div className="">ERROR : {message}</div>
        </div></div>
    </>
  );
}

export default Error;
