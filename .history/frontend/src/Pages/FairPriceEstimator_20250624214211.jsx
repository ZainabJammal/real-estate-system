import React, { useState, useEffect } from 'react';
import './FairPriceEstimator.css'; // We'll create this CSS file next

const API_URL = "http://127.0.0.1:5000"; // Your local backend URL

const PriceEstimator = () => {
  // State for the form inputs
  const [province, setProvince] = useState('');
  const [district, setDistrict] = useState('');
  const [propertyType, setPropertyType] = useState('');
  const [bedrooms, setBedrooms] = useState(0);
  const [bathrooms, setBathrooms] = useState(1);
  const [size, setSize] = useState('');

  // State to hold the configuration data from the API
  const [config, setConfig] = useState(null);
  
  // State for the UI/API interaction
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  // --- Data Fetching Effect ---
  // This runs once when the component mounts to get filter data
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch(`${API_URL}/api/v1/config/filters`);
        if (!response.ok) {
          throw new Error("Could not connect to the backend. Is the server running?");
        }
        const data = await response.json();
        setConfig(data);
      } catch (err) {
        setError(err.message);
      }
    };
    fetchConfig();
  }, []);

  // --- Event Handlers ---
  const handleProvinceChange = (e) => {
    setProvince(e.target.value);
    // Reset dependent fields
    setDistrict('');
    setPropertyType('');
    setResult(null);
  };
  
  const handleDistrictChange = (e) => {
    setDistrict(e.target.value);
    // Reset dependent fields
    setPropertyType('');
    setResult(null);
  };

  const handleTypeChange = (e) => {
    const newType = e.target.value;
    setPropertyType(newType);
    setResult(null);

    // If type is 'Land', set bedrooms and bathrooms to 0
    const typesWithoutRooms = ['Land', 'Shop', 'Warehouse', 'Factory'];
    if (typesWithoutRooms.includes(newType)) {
      setBedrooms(0);
      setBathrooms(0);
    } else {
        // Reset to default for other types
        setBedrooms(1);
        setBathrooms(1);
    }
  };

  // --- API Call to Get Estimate ---
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent form from reloading the page
    
    if (!province || !district || !propertyType || !size) {
        setError("Please fill in all fields before estimating.");
        return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    const payload = {
      province,
      district,
      type: propertyType,
      bedrooms: Number(bedrooms),
      bathrooms: Number(bathrooms),
      size_m2: Number(size),
    };

    try {
      const response = await fetch(`${API_URL}/api/v1/estimate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "An unknown error occurred.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // --- Dynamic options for dependent dropdowns ---
  const districtOptions = config && province ? config.provinces[province]?.districts || [] : [];
  const typeOptions = config && province ? config.provinces[province]?.property_types || [] : [];
  const bedroomOptions = config ? config.bedroom_options || [] : [];
  const typesWithRooms = ['Apartment', 'House/Villa', 'Chalet', 'Office', 'Residential Building'];


  // --- Render Logic ---
  return (
    <div className="estimator-container">
      <div className="estimator-card">
        <h1>Fair Price Estimator</h1>
        
        {error && <p className="error-message">{error}</p>}
        
        {!config && !error && <p>Loading configuration...</p>}
        
        {config && (
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="province">Province</label>
              <select id="province" value={province} onChange={handleProvinceChange}>
                <option value="">-- Select Province --</option>
                {Object.keys(config.provinces).map(prov => (
                  <option key={prov} value={prov}>{prov}</option>
                ))}
              </select>
            </div>
            
            {province && (
              <div className="form-group">
                <label htmlFor="district">District</label>
                <select id="district" value={district} onChange={handleDistrictChange} disabled={!province}>
                  <option value="">-- Select District --</option>
                  {districtOptions.map(dist => (
                    <option key={dist} value={dist}>{dist}</option>
                  ))}
                </select>
              </div>
            )}

            {district && (
                <div className="form-group">
                    <label htmlFor="type">Property Type</label>
                    <select id="type" value={propertyType} onChange={handleTypeChange} disabled={!district}>
                         <option value="">-- Select Type --</option>
                         {typeOptions.map(type => (
                             <option key={type} value={type}>{type}</option>
                         ))}
                    </select>
                </div>
            )}

            {propertyType && (
                <div className="extra-fields">
                    {typesWithRooms.includes(propertyType) && (
                         <div className="form-group">
                            <label htmlFor="bedrooms">Bedrooms</label>
                            <select id="bedrooms" value={bedrooms} onChange={(e) => setBedrooms(e.target.value)}>
                                {bedroomOptions.map(bed => (
                                    <option key={bed} value={bed}>{bed === 0 ? "Studio" : bed}</option>
                                ))}
                            </select>
                        </div>
                    )}
                     <div className="form-group">
                        <label htmlFor="bathrooms">Bathrooms</label>
                        <input 
                            type="number" 
                            id="bathrooms" 
                            value={bathrooms} 
                            onChange={(e) => setBathrooms(e.target.value)} 
                            min="0"
                            disabled={!typesWithRooms.includes(propertyType)}
                        />
                    </div>
                     <div className="form-group">
                        <label htmlFor="size_m2">Size (m²)</label>
                        <input 
                            type="number" 
                            id="size_m2" 
                            value={size}
                            onChange={(e) => setSize(e.target.value)}
                            placeholder="e.g., 150" 
                            required
                        />
                    </div>
                </div>
            )}
            
            <button type="submit" disabled={loading}>
              {loading ? 'Estimating...' : 'Estimate Price'}
            </button>
          </form>
        )}
        
        {loading && <div className="loader"></div>}

        {result && (
          <div className="results-card">
            <h2>Estimated Fair Price</h2>
            <p className="price">
              ${result.estimated_price.toLocaleString()}
            </p>
            <div className="range">
                Expected Range: ${result.price_range_low.toLocaleString()} - ${result.price_range_high.toLocaleString()}
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default PriceEstimator;