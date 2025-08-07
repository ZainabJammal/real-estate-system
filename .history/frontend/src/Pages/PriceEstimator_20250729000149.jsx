// import { useState, useEffect } from 'react';
// import axios from 'axios'; 
// import Autosuggest from 'react-autosuggest'; 
// import './PriceEstimator.css';

// const GEOAPIFY_API_KEY = import.meta.env.VITE_GEOAPIFY_API_KEY; 

// const PriceEstimator = () => {
//   //  console.log('VITE_GEOAPIFY_API_KEY from import.meta.env:', import.meta.env.VITE_GEOAPIFY_API_KEY);


//   const [propertyType, setPropertyType] = useState('');
//   const [province, setProvince] = useState('');
//   const [city, setCity] = useState('');
//   const [district, setDistrict] = useState('');
//   const [sizeM2, setSizeM2] = useState('');
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
//       // provinces: { 'Beirut': { districts: ['Beirut'] }, 'Mount Lebanon': { districts: ['El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley'] }, 'North': { districts: ['Batroun', 'Tripoli', 'Koura', 'Zgharta'] }, 'South': { districts: ['Saida', 'Jezzine', 'Tyre'] }, 'Bekaa': { districts: ['Zahle', 'Baalbeck', 'Rashaya'] }, 'Nabatieh': { districts: ['Nabatieh', 'Hasbaya', 'Marjeyoun'] }, },
//       // // property_types: ['Apartment', 'Office', 'Shop', 'House/Villa', 'Chalet', 'Residential Building'],
//       // bedroom_options: ['0', '1', '2', '3', '4', '5+'],
//       // bathroom_options: ['1', '2', '3', '4', '5+'],
//       provinces: { 'Beirut': { districts: ['Beirut'] , city: ['Achrafieh','Badaro','Clemenceau','Down Town','Jnah','Ras Beirut','Ras El Nabeh']}, 
//         'Mount Lebanon': { districts: ['El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley'], city:['Ain El Rimmaneh'] }, 
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

// //   const sizeCategoryOptions= [
// //     'Studio/Small (40-80)', 
// //     'Standard (81-120)', 
// //     'Comfortable (121-180)', 
// //     'Large (181-250)', 
// //     'Very Large (251-400)', 
// //     'Penthouse (401+)'
// // ]; 

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
  
//   // The handleSubmit function remains exactly the same as before.
//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!coordinates.lat) {
//       setError("Please select a valid address from the suggestions list.");
//       return;
//     }
//     setLoading(true);
//     setError('');
//     setResult(null);

//     const payload = {
//       type: propertyType, province, district,
//       sizeM2: sizeM2,
//       bedrooms: Number(bedrooms.replace('+', '')),
//       bathrooms: Number(bathrooms.replace('+', '')),
//       latitude: coordinates.lat,
//       longitude: coordinates.lng,
//     };

//     try {
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
  
//   // The entire return (...) JSX structure remains exactly the same as the Mapbox version.
//   return (
//     <div className="estimator-layout">
//       <div className="estimator-content">
//         <div className="estimator-title" >
//           <h1>The Fair Price Estimator</h1>
//         </div>
//         <div className="dashboard-components">
//           <div className="form-card">
//             {error && <p className="error-message">{error}</p>}
//             {config && (
//               <form onSubmit={handleSubmit}>
//                 <div className="form-group">
//                   <label htmlFor="propertyType">Property Type</label>
//                   <select id="propertyType" value={propertyType} onChange={(e) => setPropertyType(e.target.value)} required>
//                     <option value="">-- Select Type --</option>
//                     {config.property_types.map(type => <option key={type} value={type}>{type}</option>)}
//                   </select>
//                 </div>
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
//                 <div className="form-group">
                  
                     
//                      <label htmlFor="sizeM2">Apartment Size (in m²)</label>
//                      <div className="form-group">
//                      <input 
//                          type="number"
//                          id="sizeM2"
//                          value={sizeM2}
//                          onChange={(e) => setSizeM2(e.target.value)}
//                          placeholder="e.g., 150"
//                          min="40"
//                          max="500"
//                          required
//                   />
//                  </div>
//                 </div>
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
//                 <div>
//                   <div className="form-group" style={{ marginLeft: '50%' }}>
//                     <button type="submit" disabled={loading || !coordinates.lat}>
//                       {loading ? 'Estimating...' : 'Estimate Price'}
//                     </button>
//                   </div>
//                 </div>
//               </form>
//             )}
//             {loading && <div className="loader"></div>}
//             {result && (
//               <div className="results-card">
//                 {result.status === 'high_confidence' && (
//                   <>
//                     <h2>Estimated Fair Price</h2>
//                     <p className="price">${result.prediction.toLocaleString()}</p>
//                   </>
//                 )}
//                 {result && (
//                   <div className="results-card">
//                     <h2>Estimated Fair Price</h2>
                    
                    
//                     <p className="price">
//                       ${result.estimated_price.toLocaleString()}
//                     </p>
                    
//                     {/* Also use the correct keys for the price range */}
//                     <div className="range">
//                       Expected Range: ${result.price_range_low.toLocaleString()} - ${result.price_range_high.toLocaleString()}
//                     </div>

//                     {/* The disclaimer is fine, but you might want to use the district from the response if it's available */}
//                     {result.district && (
//                       <p className="disclaimer">
//                         This is an estimate for a typical property in the {result.district} district.
//                       </p>
//                     )}
//                   </div>
//                 )}
//               </div>
//             )}
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default PriceEstimator;
import { useState, useEffect } from 'react';
import axios from 'axios'; 
import Autosuggest from 'react-autosuggest'; 
import './PriceEstimator.css';

const GEOAPIFY_API_KEY = import.meta.env.VITE_GEOAPIFY_API_KEY; 

const PriceEstimator = () => {
  const [province, setProvince] = useState('');
  const [district, setDistrict] = useState('');
  const [city, setCity] = useState(''); 
  const [sizeM2, setSizeM2] = useState('');
  const [bedrooms, setBedrooms] = useState('2');
  const [bathrooms, setBathrooms] = useState('2');
  const [address, setAddress] = useState('');
  const [coordinates, setCoordinates] = useState({ lat: null, lng: null });
  const [suggestions, setSuggestions] = useState([]);
  const [config, setConfig] = useState(null); // Will hold our nested config
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!GEOAPIFY_API_KEY) {
      setError("FATAL: Geoapify API key is not configured.");
    }
  
    // ### --- MODIFIED: The entire nested configuration object --- ###
    const staticConfig = {
      provinces: {
        'Beirut': {
          districts: {
            'Beirut': {
              cities: ['Achrafieh', 'Badaro', 'Clemenceau', 'Down Town', 'Jnah', 'Ras Beirut', 'Ras El Nabeh']
            }
          }
        },
        'Mount Lebanon': {
          districts: {
            'Aley': { cities: ['Ain El Rimaneh'] },
            'Baabda': { cities: ['Ain Al Remmane', 'Al Jamhour', 'Baabda', 'Betchay', 'Fiyadiye', 'Furn El Chebbak', 'Hadath', 'Hazmieh', 'Louaize', 'Mar Takla', 'New Mar Takla', 'Rihaniyeh', 'Yarzeh'] },
            'El Metn': { cities: ['Aatchane', 'Ain Aar', 'Ain Najem', 'Ain Saadeh', 'Antelias', 'Aoukar', 'Baabdat', 'Beit Chabeb', 'Beit El Chaar', 'Beit El Kikko', 'Beit Meri', 'Bhorsaf', 'Biakout', 'Biyada', 'Bolonia', 'Bouchrieh', 'Broumana', 'Bsalim', 'Dahr El Sawan', 'Daychounieh', 'Dbayeh', 'Dekweneh', 'Dik El Mehdi', 'Douar', 'Elissar', 'Fanar', 'Horsh Tabet', 'Jal el Dib', 'Jdeideh', 'Jouret Al Ballout', 'Kornet Chehwan', 'Mansourieh', 'Mar Moussa', 'Mar Roukoz', 'Mazraat Yachouh', 'Monteverde', 'Mtayleb', 'Nabey', 'Naccache', 'New Rawda', 'Qannabet Broumana', 'Qornet El Hamra', 'Rabieh', 'Rawbeh', 'Rawda', 'Sabtieh', 'Sin El Fil', 'Tilal Ain Saade', 'Zalka', 'Zikrit'] },
            'Jbeil': { cities: ['Aamchit', 'Aannaya', 'Barbara', 'Bchelli', 'Bentaael', 'Blat', 'Braij', 'Edde', 'Fidar', 'Gherfine', 'Halat', 'Hboub', 'Hosrayel', 'Jbeil', 'Jdayel', 'Mastita', 'Mechmech', 'Mejdel', 'Nahr Ibrahim', 'Qartaboun'] },
            'Kesrouane': { cities: ['Achkout', 'Adma', 'Adonis', 'Ain El Rihani', 'Ajaltoun', 'Ballouneh', 'Bkerke', 'Rayfoun', 'Safra', 'Sahel Alma', 'Sarba', 'Shayle', 'Tabarja', 'Zouk Mikael', 'Zouk Mosbeh'] }
          }
        },
        'North': {
          districts: {
            'Batroun': { cities: ['Basbina', 'Batroun', 'Chekka', 'Ijdabra', 'Kfar Abida'] }
          }
        }
      },
      bedroom_options: ['1', '2', '3', '4', '5+'],
      bathroom_options: ['1', '2', '3', '4', '5+'],
    };
    setConfig(staticConfig);
  }, []);


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
 const onSuggestionSelected = (_event, { suggestion }) => {
//     // The address and coordinates are in different places in the response
//     setAddress(suggestion.properties.formatted);
//     const { lat, lon: lng } = suggestion.properties; // Geoapify uses 'lon'
//     setCoordinates({ lat, lng });
//   };

//   const getSuggestionValue = suggestion => suggestion.properties.formatted;
//   const renderSuggestion = suggestion => <div>{suggestion.properties.formatted}</div>;
  const inputProps = {
    placeholder: 'Type an address or area...',
    value: address,
    onChange: (_event, { newValue }) => setAddress(newValue),
    required: true,
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!coordinates.lat || !sizeM2 || !city) {
      setError("Please fill all fields, including size, city, and a valid address.");
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);

    const payload = {
      province, district, city,
      size_m2: Number(sizeM2),
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
  
  // ### --- MODIFIED: Logic to get dependent options --- ###
  const districtOptions = config && province ? Object.keys(config.provinces[province]?.districts || {}) : [];
  const cityOptions = config && province && district ? config.provinces[province]?.districts[district]?.cities || [] : [];
  
  return (
    <div className="estimator-layout">
      <div className="estimator-content">
        <div className="estimator-title" >
          <h1>The Fair Price Estimator</h1>
          <p>Powered by an Optimized XGBoost Model</p>
        </div>
        <div className="dashboard-components">
          <div className="form-card">
            {error && <p className="error-message">{error}</p>}
            {config && (
              <form onSubmit={handleSubmit}>
                {/* --- Province Dropdown --- */}
                <div className="form-group">
                  <label htmlFor="province">Province</label>
                  <select 
                    id="province" 
                    value={province} 
                    onChange={(e) => { 
                      setProvince(e.target.value); 
                      setDistrict(''); // Reset district on province change
                      setCity('');     // Reset city on province change
                    }} 
                    required
                  >
                    <option value="">-- Select Province --</option>
                    {Object.keys(config.provinces).map(prov => <option key={prov} value={prov}>{prov}</option>)}
                  </select>
                </div>
                
                {/* --- District Dropdown --- */}
                <div className="form-group">
                  <label htmlFor="district">District</label>
                  <select 
                    id="district" 
                    value={district} 
                    onChange={(e) => {
                      setDistrict(e.target.value);
                      setCity(''); // Reset city on district change
                    }} 
                    disabled={!province} 
                    required
                  >
                    <option value="">-- Select District --</option>
                    {districtOptions.map(dist => <option key={dist} value={dist}>{dist}</option>)}
                  </select>
                </div>
                
                {/* ### --- MODIFIED: City Dropdown --- ### */}
                <div className="form-group">
                  <label htmlFor="city">City / Town</label>
                  <select 
                    id="city"
                    value={city}
                    onChange={(e) => setCity(e.target.value)}
                    disabled={!district} // Disabled until a district is chosen
                    required
                  >
                    <option value="">-- Select City --</option>
                    {cityOptions.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>

                {/* Bedrooms and Bathrooms selectors (unchanged) */}
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

                {/* Size input (unchanged from last version) */}
                <div className="form-group">
                    <label htmlFor="sizeM2">Apartment Size (in m²)</label>
                    <input 
                        type="number" id="sizeM2" value={sizeM2}
                        onChange={(e) => setSizeM2(e.target.value)}
                        placeholder="e.g., 150" min="20" required
                    />
                </div>
                
                {/* Address Autosuggest (unchanged) */}
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
                
                <div className="form-group" style={{ textAlign: 'center' }}>
                  <button type="submit" disabled={loading}>
                    {loading ? 'Estimating...' : 'Estimate Price'}
                  </button>
                </div>
              </form>
            )}

            {/* Results display section (unchanged) */}
            {loading && <div className="loader"></div>}
            {result && (
              <div className="results-card">
                <h2>Estimated Fair Price</h2>
                <p className="price">${result.estimated_price.toLocaleString()}</p>
                <div className="range">
                  Expected Range: ${result.price_range_low.toLocaleString()} - ${result.price_range_high.toLocaleString()}
                </div>
                <p className="disclaimer">
                  This is an AI-generated estimate based on market data for a typical property with these features.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PriceEstimator;