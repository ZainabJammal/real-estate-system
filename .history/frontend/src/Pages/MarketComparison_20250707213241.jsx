import { useState, useEffect } from 'react';
import axios from 'axios'; 
import Autosuggest from 'react-autosuggest'; 
import './market.css';

const GEOAPIFY_API_KEY = import.meta.env.VITE_GEOAPIFY_API_KEY; 

const MarketComparison = () => {
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
    <div className="estimator-layout">
      <div className="estimator-content">
        <div className="estimator-title" style={{ fontSize: '20px' }}>
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
                  <label htmlFor="size_m2">Size (m²)</label>
                  <input type="number" id="size_m2" value={size} onChange={(e) => setSize(e.target.value)} placeholder="e.g., 150" min="20" required />
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
                {result && result.status === 'success' && (
                  <div className="results-card">
                    <h2>Estimated Fair Price</h2>
                    <p className="price">
                      ${result['predicted_price_$'].toLocaleString()}
                    </p>
                    <p className="disclaimer">
                      This is an estimate for a typical property in the {result.district} district.
                    </p>
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