import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

<script 
      src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=places" 
      async 
      defer
    ></script>