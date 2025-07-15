import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Not needed for the mock version
import './MarketComparison.css'; // You'll need to create/update this CSS file

// const API_URL = "http://127.0.0.1:8000"; // Your local backend URL

const PlusIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
  </svg>
);

const MarketComparison = () => {
  // State for the list of selected districts
  const [selectedDistricts, setSelectedDistricts] = useState(['', 'El Metn']); // Start with two defaults

  // State for other filters
  const [propertyType, setPropertyType] = useState('Apartment');
  
  // --- NEW: State for bedroom and bathroom filters ---
  const [selectedBedrooms, setSelectedBedrooms] = useState([2, 3]); // Default selection
  // In a real app, these options would come from the /config endpoint
  const availableBedrooms = [0, 1, 2, 3, 4, 5]; 

  // State to hold data for the dropdowns
  const [allDistricts, setAllDistricts] = useState([]);

  // State for API interaction and results
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [comparisonData, setComparisonData] = useState(null);

  useEffect(() => {
    const districtsFromData = [
      "Beirut", "El Metn", "Kesrouane", "Jbeil", "Baabda", "Aley",
      "Batroun"
    ].sort();
    setAllDistricts(districtsFromData);
  }, []);

  const handleDistrictChange = (index, value) => {
    const newSelectedDistricts = [...selectedDistricts];
    newSelectedDistricts[index] = value;
    setSelectedDistricts(newSelectedDistricts);
  };

  const handleAddDistrict = () => {
    if (selectedDistricts.length < 5) {
      setSelectedDistricts([...selectedDistricts, '']);
    }
  };

  const handleRemoveDistrict = (index) => {
    if (selectedDistricts.length > 2) {
      const newSelectedDistricts = selectedDistricts.filter((_, i) => i !== index);
      setSelectedDistricts(newSelectedDistricts);
    }
  };

  // --- NEW: Handler for bedroom checkbox changes ---
  const handleBedroomChange = (bedroom) => {
    setSelectedBedrooms(prev => 
      prev.includes(bedroom)
        ? prev.filter(b => b !== bedroom) // Uncheck: remove it
        : [...prev, bedroom] // Check: add it
    );
  };


  const handleCompare = async () => {
    const validDistricts = [...new Set(selectedDistricts.filter(d => d))];

    if (validDistricts.length < 2) {
      setError("Please select at least two different districts to compare.");
      return;
    }
    if (selectedBedrooms.length === 0) {
      setError("Please select at least one bedroom count.");
      return;
    }


    setLoading(true);
    setError('');
    setComparisonData(null);

    // --- UPDATED: Construct query parameters with new filters ---
    const params = new URLSearchParams({
        type: propertyType,
        districts: validDistricts.join(','),
        bedrooms: selectedBedrooms.join(',') // Add bedrooms to the query
    });

    try {
        const response = await fetch(`http://127.0.0.1:8000/api/v1/compare?${params.toString()}`);
        
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "An error occurred from the API.");
        }
        
        setComparisonData(data);

    } catch (error) {
        setError(error.message || "Failed to fetch comparison data. Please try again.");
    } finally {
        setLoading(false);
    }
  }
  return (
     <div className="comparison-layout">
       <div className="comparison-content">
          <div className="estimator-title">
            <h1>The Market Comparison Tool</h1>
            <p>Compare key real estate metrics across different districts for Apartments.</p>
          </div>
        
           <div className="form-card">
            <div className="form-header">
              <h3>Select Your Comparison Criteria</h3>
            </div>
            <div className="form-grid">
              <div className="form-group district-selectors">
                <label>Districts to Compare</label>
                {selectedDistricts.map((district, index) => (
                  <div key={index} className="district-input-row">
                    <select value={district} onChange={(e) => handleDistrictChange(index, e.target.value)}>
                      <option value="">-- Select District {index + 1} --</option>
                      {allDistricts.map(d => (
                        <option key={d} value={d}>{d}</option>
                      ))}
                    </select>
                    {selectedDistricts.length > 2 && (
                      <button className="remove-btn" onClick={() => handleRemoveDistrict(index)}>×</button>
                    )}
                  </div>
                ))}
                {selectedDistricts.length < 5 && (
                  <button className="add-btn" onClick={handleAddDistrict}>
                    <PlusIcon /> Add another location
                  </button>
                )}
              </div>

              {/* --- NEW: Bedroom Filter UI --- */}
              <div className="form-group bedroom-filter">
                  <label>Bedrooms</label>
                  <div className="checkbox-group">
                      {availableBedrooms.map(bed => (
                          <div key={bed} className="checkbox-wrapper">
                              <input 
                                  type="checkbox"
                                  id={`bed-${bed}`}
                                  value={bed}
                                  checked={selectedBedrooms.includes(bed)}
                                  onChange={() => handleBedroomChange(bed)}
                              />
                              <label htmlFor={`bed-${bed}`}>{bed === 0 ? "Studio" : `${bed} Bed`}</label>
                          </div>
                      ))}
                  </div>
              </div>
            </div>

            <div className="form-footer">
              <button className="compare-btn" onClick={handleCompare} disabled={loading || selectedDistricts.filter(d => d).length < 2}>
                  {loading ? 'Analyzing...' : 'Compare Markets'}
              </button>
            </div>
           </div>
          
           {error && <p className="error-message">{error}</p>}
      
           {loading && <div className="loader"></div>}

           {comparisonData && (
            <div className="results-panel">
              <h2>Comparison Results</h2>
              <div className="comparison-table-container">
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      {comparisonData.comparison_results.map(item => (
                        <th key={item.district}>{item.district}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>Median Price</strong><br/><span>Typical asking price</span></td>
                      {comparisonData.comparison_results.map(item => <td key={item.district}>${Math.round(item.median_price).toLocaleString()}</td>)}
                    </tr>
                    <tr>
                      <td><strong>Median Price per m²</strong><br/><span>The best metric for value</span></td>
                      {comparisonData.comparison_results.map(item => <td key={item.district}>${Math.round(item.median_sqm_price).toLocaleString()}</td>)}
                    </tr>
                    <tr>
                      <td><strong>Typical Size</strong><br/><span>Avg. for selected bedrooms</span></td>
                      {comparisonData.comparison_results.map(item => <td key={item.district}>{Math.round(item.typical_size)} m²</td>)}
                    </tr>
                    <tr>
                      <td><strong>Market Activity</strong><br/><span>Number of active listings</span></td>
                      {comparisonData.comparison_results.map(item => <td key={item.district}>{item.market_activity} listings</td>)}
                    </tr>
                    <tr>
                      <td><strong>Price Range</strong><br/><span>10th to 90th percentile</span></td>
                      {comparisonData.comparison_results.map(item => (
                        <td key={item.district}>
                            ${Math.round(item.price_p10/1000)}k - ${Math.round(item.price_p90/1000)}k
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
    </div>
  );
};

export default MarketComparison;