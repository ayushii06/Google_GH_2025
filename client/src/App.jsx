import './App.css';
import { Route, Routes, useNavigate } from 'react-router-dom';
import Home from './pages/Home';
import Error from './pages/Error';

function App() {

  return (
    <div className=' w-screen min-h-screen  text-white bg-black  flex flex-col font-inter' >
     
      <Routes>
      
        <Route path='/' element={<Home/>} />
      
        <Route path='*' element={<Error />} />

      </Routes>
    </div>
  );
}

export default App;
