import React, { useState, useEffect } from 'react';

import PlacesAutocomplete, { geocodeByAddress, getLatLng } from 'react-places-autocomplete';
import './FairPriceEstimator.css';

// *** BEST PRACTICE: Define your API URL once ***
// const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000'; // Or your Quart server's port

const PriceEstimator = () => {
  // --- Form State ---
  const [propertyType, setPropertyType] = useState('');
  const [province, setProvince] = useState('');
  const [district, setDistrict] = useState('');
  const [size, setSize] = useState('');
  const [bedrooms, setBedrooms] = useState('2'); // Default to a common value
  const [bathrooms, setBathrooms] = useState('2');
  
  // *** NEW: State for location input ***
  const [address, setAddress] = useState('');
  const [coordinates, setCoordinates] = useState({ lat: null, lng: null });

  // --- UI/API State ---
  const [config, setConfig] = useState(null); // Holds dropdown options
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  // --- Data Fetching for Dropdowns (runs once) ---
  useEffect(() => {
    // In a real app, you'd fetch this from a /config endpoint.
    // For now, we can hardcode it based on our knowledge of the data.
    const staticConfig = {
      provinces: {
        'Beirut': { districts: ['Beirut'] },
        'Mount Lebanon': { districts: ['El Metn', 'Kesrouane', 'Jbeil', 'Baabda', 'Aley', 'Chouf'] },
        'North': { districts: ['Batroun', 'Tripoli', 'Koura', 'Zgharta'] },
        // ... add other provinces and districts as needed
      },
      property_types: ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Shop', 'Residential Building'],
      bedroom_options: ['0', '1', '2', '3', '4', '5+'],
      bathroom_options: ['1', '2', '3', '4', '5+'],
    };
    setConfig(staticConfig);
  }, []);

  // --- Location Handlers ---
  const handleAddressSelect = async (selectedAddress) => {
    setAddress(selectedAddress);
    try {
      const results = await geocodeByAddress(selectedAddress);
      const latLng = await getLatLng(results[0]);
      setCoordinates(latLng);
      console.log('Selected Coords:', latLng);
    } catch (err) {
      setError('Could not get coordinates for the selected address.');
      console.error('Geocoding Error:', err);
    }
  };

  // --- Form Submission ---
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!propertyType || !province || !district || !size || !coordinates.lat) {
      setError("Please fill all fields, including a valid address.");
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    // This is the exact object our backend /predict_property endpoint expects
    const payload = {
      type: propertyType,
      province,
      district,
      size_m2: Number(size),
      bedrooms: Number(bedrooms),
      bathrooms: Number(bathrooms),
      latitude: coordinates.lat,
      longitude: coordinates.lng,
    };

    try {
      const response = await fetch(`${API_URL}/predict_property`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "An unknown error occurred from the server.");
      }
      setResult(data); // The result will have a 'status' key
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const districtOptions = config && province ? config.provinces[province]?.districts || [] : [];
  
  return (
    <div className="estimator-container">
      <div className="estimator-card">
        <h1>Fair Price Estimator</h1>
        <p className="subtitle">Get an instant, data-driven valuation for properties in Lebanon.</p>

        {error && <p className="error-message">{error}</p>}
        
        {!config && !error && <p>Loading...</p>}
        
        {config && (
          <form onSubmit={handleSubmit}>
            {/* --- CORE INPUTS --- */}
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
                    {config.bedroom_options.map(val => <option key={val} value={val}>{val}</option>)}
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
                <input 
                  type="number" id="size_m2" value={size}
                  onChange={(e) => setSize(e.target.value)}
                  placeholder="e.g., 150" min="20" required
                />
              </div>
            </div>

            {/* --- ADDRESS INPUT (Required for lat/lon) --- */}
            <div className="form-group address-group">
              <label htmlFor="address">Address or Area</label>
              <PlacesAutocomplete
                value={address}
                onChange={setAddress}
                onSelect={handleAddressSelect}
              >
                {({ getInputProps, suggestions, getSuggestionItemProps, loading }) => (
                  <div>
                    <input {...getInputProps({ placeholder: 'Type address...' })} required />
                    <div className="autocomplete-dropdown-container">
                      {loading && <div>Loading...</div>}
                      {suggestions.map(suggestion => {
                        const style = suggestion.active ? { backgroundColor: '#fafafa', cursor: 'pointer' } : { backgroundColor: '#ffffff', cursor: 'pointer' };
                        return (
                          <div {...getSuggestionItemProps(suggestion, { style })}>
                            <span>{suggestion.description}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </PlacesAutocomplete>
            </div>
            
            <button type="submit" disabled={loading || !coordinates.lat}>
              {loading ? 'Estimating...' : 'Estimate Price'}
            </button>
          </form>
        )}
        
        {/* --- DYNAMIC RESULTS DISPLAY --- */}
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