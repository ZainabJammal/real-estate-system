import { useState, useEffect } from 'react';
import axios from 'axios'; 
import Autosuggest from 'react-autosuggest'; 
import './PriceEstimator.css';

const GEOAPIFY_API_KEY = import.meta.env.VITE_GEOAPIFY_API_KEY; 

const PriceEstimator = () => {
  //  console.log('VITE_GEOAPIFY_API_KEY from import.meta.env:', import.meta.env.VITE_GEOAPIFY_API_KEY);


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
  
    const staticConfig = {
      // provinces: { 'Beirut': { districts: ['Beirut'] }, 'Mount Lebanon': { districts: ['El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley'] }, 'North': { districts: ['Batroun', 'Tripoli', 'Koura', 'Zgharta'] }, 'South': { districts: ['Saida', 'Jezzine', 'Tyre'] }, 'Bekaa': { districts: ['Zahle', 'Baalbeck', 'Rashaya'] }, 'Nabatieh': { districts: ['Nabatieh', 'Hasbaya', 'Marjeyoun'] }, },
      // // property_types: ['Apartment', 'Office', 'Shop', 'House/Villa', 'Chalet', 'Residential Building'],
      // bedroom_options: ['0', '1', '2', '3', '4', '5+'],
      // bathroom_options: ['1', '2', '3', '4', '5+'],
      provinces: { 'Beirut': { districts: ['Beirut'] }, 
        'Mount Lebanon': { districts: ['El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley'] }, 
        'North': { districts: ['Batroun'] } },
      property_types: ['Apartment'],
      bedroom_options: ['1', '2', '3', '4'],
      bathroom_options: ['1', '2', '3', '4'],
//       sizeCategoryOptions: [
//     'Studio/Small (40-80)', 
//     'Standard (81-120)', 
//     'Comfortable (121-180)', 
//     'Large (181-250)', 
//     'Very Large (251-400)', 
//     'Penthouse (401+)'
// ],
     

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

//   const sizeCategoryOptions= [
//     'Studio/Small (40-80)', 
//     'Standard (81-120)', 
//     'Comfortable (121-180)', 
//     'Large (181-250)', 
//     'Very Large (251-400)', 
//     'Penthouse (401+)'
// ]; 

  const onSuggestionSelected = (_event, { suggestion }) => {
    // The address and coordinates are in different places in the response
    setAddress(suggestion.properties.formatted);
    const { lat, lon: lng } = suggestion.properties; // Geoapify uses 'lon'
    setCoordinates({ lat, lng });
  };

  const getSuggestionValue = suggestion => suggestion.properties.formatted;
  const renderSuggestion = suggestion => <div>{suggestion.properties.formatted}</div>;

  const inputProps = {
    placeholder: 'Type an address or area...',
    value: address,
    onChange: (_event, { newValue }) => setAddress(newValue),
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
      size: size,
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
    <div className="estimator-layout">
      <div className="estimator-content">
        <div className="estimator-title" >
          <h1>The Fair Price Estimator</h1>
        </div>
        <div className="dashboard-components">
          <div className="form-card">
            {error && <p className="error-message">{error}</p>}
            {config && (
              <form onSubmit={handleSubmit}>
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
                  
                     
                     <label htmlFor="sizeM2">Apartment Size (in m²)</label>
                     <input 
                         type="number"
                         id="sizeM2"
                         value={size}
                         onChange={(e) => setSize(e.target.value)}
                         placeholder="e.g., 150"
                         min="40"
                         max="500"
                         required
                  />
                 </div>
                </div>
                <div className="area-address-group">
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
                <div>
                  <div className="form-group" style={{ marginLeft: '50%' }}>
                    <button type="submit" disabled={loading || !coordinates.lat}>
                      {loading ? 'Estimating...' : 'Estimate Price'}
                    </button>
                  </div>
                </div>
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
                {result && (
                  <div className="results-card">
                    <h2>Estimated Fair Price</h2>
                    
                    
                    <p className="price">
                      ${result.estimated_price.toLocaleString()}
                    </p>
                    
                    {/* Also use the correct keys for the price range */}
                    <div className="range">
                      Expected Range: ${result.price_range_low.toLocaleString()} - ${result.price_range_high.toLocaleString()}
                    </div>

                    {/* The disclaimer is fine, but you might want to use the district from the response if it's available */}
                    {result.district && (
                      <p className="disclaimer">
                        This is an estimate for a typical property in the {result.district} district.
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PriceEstimator;

// import { useState, useEffect } from 'react';
// import axios from 'axios'; 
// import Autosuggest from 'react-autosuggest'; 
// import './PriceEstimator.css';

// const GEOAPIFY_API_KEY = import.meta.env.VITE_GEOAPIFY_API_KEY; 

// const PriceEstimator = () => {
  
//   const [propertyType, setPropertyType] = useState('');
//   const [province, setProvince] = useState('');
//   const [district, setDistrict] = useState('');
//   const [size, setSize] = useState('');
//   const [bedrooms, setBedrooms] = useState('2');
//   const [bathrooms, setBathrooms] = useState('2');
//   const [address, setAddress] = useState('');
//   const [coordinates, setCoordinates] = useState({ lat: null, lng: null });
//   const [suggestions, setSuggestions] = useState([]);
//   const [config, setConfig] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');
//   const [result, setResult] = useState(null);

//   useEffect(() => {
//     if (!GEOAPIFY_API_KEY) {
//       setError("FATAL: Geoapify API key is not configured. Please check your .env file.");
//     }
  
//     const staticConfig = {
//       provinces: { 
//         'Beirut': { districts: ['Beirut'] }, 
//         'Mount Lebanon': { districts: ['El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley'] }, 
//         'North': { districts: ['Batroun', 'Koura', 'Zgharta'] } 
//       },
//       bedroom_options: ['1', '2', '3', '4', '5+'],
//       bathroom_options: ['1', '2', '3', '4', '5+'],
//     };
//     setConfig(staticConfig);
//   }, []);

 
 
//     const staticConfig = {
//       // provinces: { 'Beirut': { districts: ['Beirut'] }, 'Mount Lebanon': { districts: ['El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley'] }, 'North': { districts: ['Batroun', 'Tripoli', 'Koura', 'Zgharta'] }, 'South': { districts: ['Saida', 'Jezzine', 'Tyre'] }, 'Bekaa': { districts: ['Zahle', 'Baalbeck', 'Rashaya'] }, 'Nabatieh': { districts: ['Nabatieh', 'Hasbaya', 'Marjeyoun'] }, },
//       // // property_types: ['Apartment', 'Office', 'Shop', 'House/Villa', 'Chalet', 'Residential Building'],
//       // bedroom_options: ['0', '1', '2', '3', '4', '5+'],
//       // bathroom_options: ['1', '2', '3', '4', '5+'],
//       provinces: { 'Beirut': { districts: ['Beirut'] }, 
//         'Mount Lebanon': { districts: ['El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley'] }, 
//         'North': { districts: ['Batroun'] } },
//       property_types: ['Apartment'],
//       bedroom_options: ['1', '2', '3', '4'],
//       bathroom_options: ['1', '2', '3', '4'],
// //       sizeCategoryOptions: [
// //     'Studio/Small (40-80)', 
// //     'Standard (81-120)', 
// //     'Comfortable (121-180)', 
// //     'Large (181-250)', 
// //     'Very Large (251-400)', 
// //     'Penthouse (401+)'
// // ],
     

//     };
//     setConfig(staticConfig);
//   }, []);

//   // --- *** GEOAPIFY AUTOSUGGEST HANDLERS (The only part that changes) *** ---
//   const onSuggestionsFetchRequested = async ({ value }) => {
//     if (value.length > 2) { // Only search after 3+ characters
//       try {
//         const response = await axios.get(
//           `https://api.geoapify.com/v1/geocode/autocomplete`, {
//             params: {
//               text: value,
//               apiKey: GEOAPIFY_API_KEY,
//               filter: 'countrycode:lb', // Restrict search to Lebanon
//               limit: 5,
//             }
//           }
//         );
//         // The data structure is slightly different from Mapbox
//         setSuggestions(response.data.features);
//       } catch (err) {
//         console.error("Geoapify API error:", err);
//       }
//     }
//   };

//   const onSuggestionsClearRequested = () => {
//     setSuggestions([]);
//   };

//   const sizeCategoryOptions= [
//     'Studio/Small (40-80)', 
//     'Standard (81-120)', 
//     'Comfortable (121-180)', 
//     'Large (181-250)', 
//     'Very Large (251-400)', 
//     'Penthouse (401+)'
// ]; 

//   const onSuggestionSelected = (_event, { suggestion }) => {
//     // The address and coordinates are in different places in the response
//     setAddress(suggestion.properties.formatted);
//     const { lat, lon: lng } = suggestion.properties; // Geoapify uses 'lon'
//     setCoordinates({ lat, lng });
//   };

//   const getSuggestionValue = suggestion => suggestion.properties.formatted;
//   const renderSuggestion = suggestion => <div>{suggestion.properties.formatted}</div>;

//   const inputProps = {
//     placeholder: 'Type an address or area...',
//     value: address,
//     onChange: (_event, { newValue }) => setAddress(newValue),
//     required: true,
//   };
  
//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!coordinates.lat || !sizeM2 || !city) {
//       setError("Please fill all fields, including size, city, and a valid address.");
//       return;
//     }
//     setLoading(true);
//     setError('');
//     setResult(null);

//     // ### --- MODIFIED: Update the payload to match the new backend requirements --- ###
//     const payload = {
//       province, 
//       district,
//       city, // Send the city
//       size_m2: Number(sizeM2), // Send the raw size_m2
//       bedrooms: Number(bedrooms.replace('+', '')),
//       bathrooms: Number(bathrooms.replace('+', '')),
//       latitude: coordinates.lat,
//       longitude: coordinates.lng,
//     };

//     try {
//       // The endpoint URL remains the same
//       const response = await fetch(`http://127.0.0.1:8000/estimate_fair_price`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(payload),
//       });
//       const data = await response.json();
//       if (!response.ok) throw new Error(data.error || "An unknown server error occurred.");
//       setResult(data);
//     } catch (err) {
//       setError(err.message);
//     } finally {
//       setLoading(false);
//     }
//   };
  
//   const districtOptions = config && province ? config.provinces[province]?.districts || [] : [];
  
//   return (
//     <div className="estimator-layout">
//       <div className="estimator-content">
//         <div className="estimator-title" >
//           <h1>The Fair Price Estimator</h1>
//           <p>Powered by an Optimized XGBoost Model</p>
//         </div>
//         <div className="dashboard-components">
//           <div className="form-card">
//             {error && <p className="error-message">{error}</p>}
//             {config && (
//               <form onSubmit={handleSubmit}>
//                 {/* --- Property Type is now fixed to Apartment, so we can hide it --- */}
//                 {/* <div className="form-group"> ... </div> */}

//                 {/* Province and District selectors (unchanged) */}
//                 <div className="form-group">
//                   <label htmlFor="province">Province</label>
//                   <select id="province" value={province} onChange={(e) => { setProvince(e.target.value); setDistrict(''); }} required>
//                     <option value="">-- Select Province --</option>
//                     {Object.keys(config.provinces).map(prov => <option key={prov} value={prov}>{prov}</option>)}
//                   </select>
//                 </div>
//                 <div className="form-group">
//                   <label htmlFor="district">District</label>
//                   <select id="district" value={district} onChange={(e) => setDistrict(e.target.value)} disabled={!province} required>
//                     <option value="">-- Select District --</option>
//                     {districtOptions.map(dist => <option key={dist} value={dist}>{dist}</option>)}
//                   </select>
//                 </div>
                
//                 {/* ### --- MODIFIED: Add City input --- ### */}
//                 <div className="form-group">
//                     <label htmlFor="city">City / Town</label>
//                     <input 
//                         type="text"
//                         id="city"
//                         value={city}
//                         onChange={(e) => setCity(e.target.value)}
//                         placeholder="e.g., Achrafieh"
//                         required
//                     />
//                 </div>

//                 {/* Bedrooms and Bathrooms selectors (unchanged) */}
//                 <div className="form-group">
//                   <label htmlFor="bedrooms">Bedrooms</label>
//                   <select id="bedrooms" value={bedrooms} onChange={(e) => setBedrooms(e.target.value)}>
//                     {config.bedroom_options.map(val => <option key={val} value={val}>{val === '0' ? 'Studio' : val}</option>)}
//                   </select>
//                 </div>
//                 <div className="form-group">
//                   <label htmlFor="bathrooms">Bathrooms</label>
//                   <select id="bathrooms" value={bathrooms} onChange={(e) => setBathrooms(e.target.value)}>
//                     {config.bathroom_options.map(val => <option key={val} value={val}>{val}</option>)}
//                   </select>
//                 </div>

//                 {/* ### --- MODIFIED: Replace Size Category with Size (m²) input --- ### */}
//                 <div className="form-group">
//                     <label htmlFor="sizeM2">Apartment Size (in m²)</label>
//                     <input 
//                         type="number"
//                         id="sizeM2"
//                         value={sizeM2}
//                         onChange={(e) => setSizeM2(e.target.value)}
//                         placeholder="e.g., 150"
//                         min="20"
//                         required
//                     />
//                 </div>
                
//                 {/* Address Autosuggest (unchanged) */}
//                 <div className="area-address-group">
//                   <label htmlFor="address">Address or Area</label>
//                   <Autosuggest
//                     suggestions={suggestions}
//                     onSuggestionsFetchRequested={onSuggestionsFetchRequested}
//                     onSuggestionsClearRequested={onSuggestionsClearRequested}
//                     onSuggestionSelected={onSuggestionSelected}
//                     getSuggestionValue={getSuggestionValue}
//                     renderSuggestion={renderSuggestion}
//                     inputProps={inputProps}
//                   />
//                 </div>
                
//                 <div className="form-group" style={{ textAlign: 'center' }}>
//                   <button type="submit" disabled={loading}>
//                     {loading ? 'Estimating...' : 'Estimate Price'}
//                   </button>
//                 </div>
//               </form>
//             )}

//             {/* --- Results display section (unchanged) --- */}
//             {loading && <div className="loader"></div>}
//             {result && (
//               <div className="results-card">
//                 <h2>Estimated Fair Price</h2>
//                 <p className="price">
//                   ${result.estimated_price.toLocaleString()}
//                 </p>
//                 <div className="range">
//                   Expected Range: ${result.price_range_low.toLocaleString()} - ${result.price_range_high.toLocaleString()}
//                 </div>
//                 <p className="disclaimer">
//                   This is an AI-generated estimate based on market data for a typical property with these features.
//                 </p>
//               </div>
//             )}
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// )};

// export default PriceEstimator;