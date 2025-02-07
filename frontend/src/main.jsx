import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { Main } from './components/main/Main'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
    <Main/>
  </StrictMode>,
)
