import { useState, useEffect } from "react";
import { DotLottieReact } from '@lottiefiles/dotlottie-react';


export default function LoadingScreen() {

  return (
    <div className="bg-black mx-auto my-20 text-white px-10 py-10">
      <DotLottieReact
      className='w-80 my-10 mx-auto font-bold'
        src="https://lottie.host/0fca43e2-e21f-4aa2-975b-b2ea83802d2f/ed9UbP7eae.lottie"
        background="transparent"
        speed="1"
        style="width: 300px; height: 300px"
        loop
        autoplay
      ></DotLottieReact>
      <div className="text-center mt-12 mb-5 font-medium text-2xl ">
        <div className="">Please wait for some time
          while we are processing the Report</div>

      </div>
    </div>
  );
}
