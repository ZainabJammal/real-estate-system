// import React, { useState, useEffect } from 'react';
// import PlacesAutocomplete, { geocodeByAddress, getLatLng } from 'react-places-autocomplete';
// // import './PriceEstimator.css';


// const PriceEstimator = () => {
//   // --- Form State ---
//   const [propertyType, setPropertyType] = useState('');
//   const [province, setProvince] = useState('');
//   const [district, setDistrict] = useState('');
//   const [size, setSize] = useState('');
//   const [bedrooms, setBedrooms] = useState('2'); // Default to a common value
//   const [bathrooms, setBathrooms] = useState('2');
  
//   // *** NEW: State for location input ***
//   const [address, setAddress] = useState('');
//   const [coordinates, setCoordinates] = useState({ lat: null, lng: null });

//   // --- UI/API State ---
//   const [config, setConfig] = useState(null); // Holds dropdown options
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');
//   const [result, setResult] = useState(null);

//   // --- Data Fetching for Dropdowns (runs once) ---
//   useEffect(() => {
//     // In a real app, you'd fetch this from a /config endpoint.
//     // For now, we can hardcode it based on our knowledge of the data.
//     const staticConfig = {
//       provinces: {
//         'Beirut': { districts: ['Beirut'] },
//         'Mount Lebanon': { districts: ['El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley', 'Chouf'] },
//         'North': { districts: ['Batroun', 'Tripoli', 'Koura', 'Zgharta'] },
//         // ... add other provinces and districts as needed
//       },
//       property_types: ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Shop'],
//       bedroom_options: ['0', '1', '2', '3', '4', '5+'],
//       bathroom_options: ['1', '2', '3', '4', '5+'],
//     };
//     setConfig(staticConfig);
//   }, []);

//   // --- Location Handlers ---
//   const handleAddressSelect = async (selectedAddress) => {
//     setAddress(selectedAddress);
//     try {
//       const results = await geocodeByAddress(selectedAddress);
//       const latLng = await getLatLng(results[0]);
//       setCoordinates(latLng);
//       console.log('Selected Coords:', latLng);
//     } catch (err) {
//       setError('Could not get coordinates for the selected address.');
//       console.error('Geocoding Error:', err);
//     }
//   };

//   // --- Form Submission ---
//   const handleSubmit = async (e) => {
//     e.preventDefault();
    
//     if (!propertyType || !province || !district || !size || !coordinates.lat) {
//       setError("Please fill all fields, including a valid address.");
//       return;
//     }

//     setLoading(true);
//     setError('');
//     setResult(null);

//     // This is the exact object our backend /predict_property endpoint expects
//     const payload = {
//       type: propertyType,
//       province,
//       district,
//       size_m2: Number(size),
//       bedrooms: Number(bedrooms),
//       bathrooms: Number(bathrooms),
//       latitude: coordinates.lat,
//       longitude: coordinates.lng,
//     };

//     try {
//       const response = await fetch(`http://127.0.0.1:8000/predict_price`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(payload),
//       });
//       const data = await response.json();
//       if (!response.ok) {
//         throw new Error(data.error || "An unknown error occurred from the server.");
//       }
//       setResult(data); // The result will have a 'status' key
//     } catch (err) {
//       setError(err.message);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const districtOptions = config && province ? config.provinces[province]?.districts || [] : [];
  
//   return (
//     <div className="estimator-container">
//       <div className="estimator-card">
//         <h1>Fair Price Estimator</h1>
//         <p className="subtitle">Get an instant, data-driven valuation for properties in Lebanon.</p>

//         {error && <p className="error-message">{error}</p>}
        
//         {!config && !error && <p>Loading...</p>}
        
//         {config && (
//           <form onSubmit={handleSubmit}>
//             {/* --- CORE INPUTS --- */}
//             <div className="form-grid">
//               <div className="form-group">
//                 <label htmlFor="propertyType">Property Type</label>
//                 <select id="propertyType" value={propertyType} onChange={(e) => setPropertyType(e.target.value)} required>
//                   <option value="">-- Select Type --</option>
//                   {config.property_types.map(type => <option key={type} value={type}>{type}</option>)}
//                 </select>
//               </div>

//               <div className="form-group">
//                 <label htmlFor="province">Province</label>
//                 <select id="province" value={province} onChange={(e) => { setProvince(e.target.value); setDistrict(''); }} required>
//                   <option value="">-- Select Province --</option>
//                   {Object.keys(config.provinces).map(prov => <option key={prov} value={prov}>{prov}</option>)}
//                 </select>
//               </div>

//               <div className="form-group">
//                 <label htmlFor="district">District</label>
//                 <select id="district" value={district} onChange={(e) => setDistrict(e.target.value)} disabled={!province} required>
//                   <option value="">-- Select District --</option>
//                   {districtOptions.map(dist => <option key={dist} value={dist}>{dist}</option>)}
//                 </select>
//               </div>

//               <div className="form-group">
//                 <label htmlFor="bedrooms">Bedrooms</label>
//                 <select id="bedrooms" value={bedrooms} onChange={(e) => setBedrooms(e.target.value)}>
//                     {config.bedroom_options.map(val => <option key={val} value={val}>{val}</option>)}
//                 </select>
//               </div>

//               <div className="form-group">
//                 <label htmlFor="bathrooms">Bathrooms</label>
//                 <select id="bathrooms" value={bathrooms} onChange={(e) => setBathrooms(e.target.value)}>
//                     {config.bathroom_options.map(val => <option key={val} value={val}>{val}</option>)}
//                 </select>
//               </div>

//               <div className="form-group">
//                 <label htmlFor="size_m2">Size (m²)</label>
//                 <input 
//                   type="number" id="size_m2" value={size}
//                   onChange={(e) => setSize(e.target.value)}
//                   placeholder="e.g., 150" min="20" required
//                 />
//               </div>
//             </div>

//             {/* --- ADDRESS INPUT (Required for lat/lon) --- */}
//             <div className="form-group address-group">
//               <label htmlFor="address">Address or Area</label>
//               <PlacesAutocomplete
//                 value={address}
//                 onChange={setAddress}
//                 onSelect={handleAddressSelect}
//               >
//                 {({ getInputProps, suggestions, getSuggestionItemProps, loading }) => (
//                   <div>
//                     <input {...getInputProps({ placeholder: 'Type address...' })} required />
//                     <div className="autocomplete-dropdown-container">
//                       {loading && <div>Loading...</div>}
//                       {suggestions.map(suggestion => {
//                         const style = suggestion.active ? { backgroundColor: '#fafafa', cursor: 'pointer' } : { backgroundColor: '#ffffff', cursor: 'pointer' };
//                         return (
//                           <div {...getSuggestionItemProps(suggestion, { style })}>
//                             <span>{suggestion.description}</span>
//                           </div>
//                         );
//                       })}
//                     </div>
//                   </div>
//                 )}
//               </PlacesAutocomplete>
//             </div>
            
//             <button type="submit" disabled={loading || !coordinates.lat}>
//               {loading ? 'Estimating...' : 'Estimate Price'}
//             </button>
//           </form>
//         )}
        
//         {/* --- DYNAMIC RESULTS DISPLAY --- */}
//         {result && (
//           <div className="results-card">
//             {result.status === 'high_confidence' && (
//               <>
//                 <h2>Estimated Fair Price</h2>
//                 <p className="price">${result.prediction.toLocaleString()}</p>
//               </>
//             )}
//             {result.status === 'low_confidence_range' && (
//               <>
//                 <h2>Estimated Price Range</h2>
//                 <p className="price">
//                   ${result.estimated_range[0].toLocaleString()} - ${result.estimated_range[1].toLocaleString()}
//                 </p>
//                 <p className="disclaimer">
//                   Prices for this property type can vary greatly based on specific features like view and finishing quality. We recommend consulting an agent for a precise valuation.
//                 </p>
//               </>
//             )}
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

// export default PriceEstimator;

import React, { useState, useEffect } from 'react';
import axios from 'axios'; // For making API calls
import Autosuggest from 'react-autosuggest'; // Our autocomplete component
import './PriceEstimator.css';
// import './MapboxAutosuggest.css'; // We can reuse the same CSS for the suggestions

// const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
// *** IMPORTANT: Store your Geoapify key in a .env file ***
const GEOAPIFY_API_KEY = import.meta.env.VITE_GEOAPIFY_API_KEY; // Your key from Geoapify

const PriceEstimator = () => {
  //  console.log('VITE_GEOAPIFY_API_KEY from import.meta.env:', import.meta.env.VITE_GEOAPIFY_API_KEY);
  // All your existing state variables remain the same...
  const [propertyType, setPropertyType] = useState('');
  const [province, setProvince] = useState('');
  const [district, setDistrict] = useState('');
  const [size, setSize] = useState('');
  const [bedrooms, setBedrooms] = useState('2');
  const [bathrooms, setBathrooms] = useState('2');
  const [address, setAddress] = useState('');
  const [coordinates, setCoordinates] = useState({ lat: null, lng: null });
  const [suggestions, setSuggestions] = useState([]);
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!GEOAPIFY_API_KEY) {
      setError("FATAL: Geoapify API key is not configured. Please check your .env file.");
    }
    // ... your config setup logic remains the same ...
    const staticConfig = {
      provinces: { 'Beirut': { districts: ['Beirut'] }, 'Mount Lebanon': { districts: ['El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley', 'Chouf'] }, 'North': { districts: ['Batroun', 'Tripoli', 'Koura', 'Zgharta'] }, 'South': { districts: ['Saida', 'Jezzine', 'Tyre'] }, 'Bekaa': { districts: ['Zahle', 'Baalbeck', 'Rashaya'] }, 'Nabatieh': { districts: ['Nabatieh', 'Hasbaya', 'Marjeyoun'] }, },
      property_types: ['Apartment', 'Office', 'Shop', 'House/Villa', 'Chalet', 'Residential Building'],
      bedroom_options: ['0', '1', '2', '3', '4', '5+'],
      bathroom_options: ['1', '2', '3', '4', '5+'],
    };
    setConfig(staticConfig);
  }, []);

  // --- *** GEOAPIFY AUTOSUGGEST HANDLERS (The only part that changes) *** ---
  const onSuggestionsFetchRequested = async ({ value }) => {
    if (value.length > 2) { // Only search after 3+ characters
      try {
        const response = await axios.get(
          `https://api.geoapify.com/v1/geocode/autocomplete`, {
            params: {
              text: value,
              apiKey: GEOAPIFY_API_KEY,
              filter: 'countrycode:lb', // Restrict search to Lebanon
              limit: 5,
            }
          }
        );
        // The data structure is slightly different from Mapbox
        setSuggestions(response.data.features);
      } catch (err) {
        console.error("Geoapify API error:", err);
      }
    }
  };

  const onSuggestionsClearRequested = () => {
    setSuggestions([]);
  };

  const onSuggestionSelected = (event, { suggestion }) => {
    // The address and coordinates are in different places in the response
    setAddress(suggestion.properties.formatted);
    const { lat, lon: lng } = suggestion.properties; // Geoapify uses 'lon'
    setCoordinates({ lat, lng });
  };

  const getSuggestionValue = suggestion => suggestion.properties.formatted;
  const renderSuggestion = suggestion => <div>{suggestion.properties.formatted}</div>;

  const inputProps = {
    placeholder: 'Start typing an address or area...',
    value: address,
    onChange: (event, { newValue }) => setAddress(newValue),
    required: true,
  };
  
  // The handleSubmit function remains exactly the same as before.
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!coordinates.lat) {
      setError("Please select a valid address from the suggestions list.");
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);

    const payload = {
      type: propertyType, province, district,
      size_m2: Number(size),
      bedrooms: Number(bedrooms.replace('+', '')),
      bathrooms: Number(bathrooms.replace('+', '')),
      latitude: coordinates.lat,
      longitude: coordinates.lng,
    };

    try {
      const response = await fetch(`http://127.0.0.1:8000/estimate_fair_price`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "An unknown server error occurred.");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const districtOptions = config && province ? config.provinces[province]?.districts || [] : [];
  
  // The entire return (...) JSX structure remains exactly the same as the Mapbox version.
  return (
    <div>
    <div className="estimator-container">
      <div className="estimator-card">
       
        
        {error && <p className="error-message">{error}</p>}
        {config && (
          <form onSubmit={handleSubmit}>
            {/* The form grid with dropdowns remains the same */}
            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="propertyType">Property Type</label>
                <select id="propertyType" value={propertyType} onChange={(e) => setPropertyType(e.target.value)} required>
                  <option value="">-- Select Type --</option>
                  {config.property_types.map(type => <option key={type} value={type}>{type}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="province">Province</label>
                <select id="province" value={province} onChange={(e) => { setProvince(e.target.value); setDistrict(''); }} required>
                  <option value="">-- Select Province --</option>
                  {Object.keys(config.provinces).map(prov => <option key={prov} value={prov}>{prov}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="district">District</label>
                <select id="district" value={district} onChange={(e) => setDistrict(e.target.value)} disabled={!province} required>
                  <option value="">-- Select District --</option>
                  {districtOptions.map(dist => <option key={dist} value={dist}>{dist}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="bedrooms">Bedrooms</label>
                <select id="bedrooms" value={bedrooms} onChange={(e) => setBedrooms(e.target.value)}>
                    {config.bedroom_options.map(val => <option key={val} value={val}>{val === '0' ? 'Studio' : val}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="bathrooms">Bathrooms</label>
                <select id="bathrooms" value={bathrooms} onChange={(e) => setBathrooms(e.target.value)}>
                    {config.bathroom_options.map(val => <option key={val} value={val}>{val}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="size_m2">Size (m²)</label>
                <input type="number" id="size_m2" value={size} onChange={(e) => setSize(e.target.value)} placeholder="e.g., 150" min="20" required />
              </div>
            </div>
            
            {/* The Autosuggest component remains the same, but now it's powered by Geoapify */}
            <div className="form-group address-group">
              <label htmlFor="address">Address or Area</label>
              <Autosuggest
                suggestions={suggestions}
                onSuggestionsFetchRequested={onSuggestionsFetchRequested}
                onSuggestionsClearRequested={onSuggestionsClearRequested}
                onSuggestionSelected={onSuggestionSelected}
                getSuggestionValue={getSuggestionValue}
                renderSuggestion={renderSuggestion}
                inputProps={inputProps}
              />
            </div>
            <button type="submit" disabled={loading || !coordinates.lat}>
              {loading ? 'Estimating...' : 'Estimate Price'}
            </button>
          </form>
        )}
        {loading && <div className="loader"></div>}
        {result && (
            <div className="results-card">
            {result.status === 'high_confidence' && (
              <>
                <h2>Estimated Fair Price</h2>
                <p className="price">${result.prediction.toLocaleString()}</p>
              </>
            )}
            {result.status === 'low_confidence_range' && (
              <>
                <h2>Estimated Price Range</h2>
                <p className="price">
                  ${result.estimated_range[0].toLocaleString()} - ${result.estimated_range[1].toLocaleString()}
                </p>
                <p className="disclaimer">
                  Prices for this property type can vary greatly based on specific features like view and finishing quality. We recommend consulting an agent for a precise valuation.
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default PriceEstimator;