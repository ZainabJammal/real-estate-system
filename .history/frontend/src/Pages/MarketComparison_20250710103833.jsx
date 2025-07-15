import React, { useState, useEffect } from 'react';
import './MarketComparison.css';

const PlusIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
  </svg>
);

// This component now expects raw numbers and will format them itself.
const MarketComparison = () => {
  const [selectedDistricts, setSelectedDistricts] = useState(['', '']);
  const [allDistricts, setAllDistricts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  // Let's store the results directly in the state
  const [comparisonResults, setComparisonResults] = useState(null);

  useEffect(() => {
    // In a real app, this list might also come from an API endpoint
    const districtsFromData = [
      "Beirut", "El Metn", "Kesrouane", "Jbeil", "Baabda", "Aley", "Batroun"
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

  const handleCompare = async () => {
    // Use a Set to get unique, non-empty districts
    const validDistricts = [...new Set(selectedDistricts.filter(d => d))];

    if (validDistricts.length < 2) {
      setError("Please select at least two different districts to compare.");
      return;
    }

    setLoading(true);
    setError('');
    setComparisonResults(null);

    const params = new URLSearchParams({
        districts: validDistricts.join(','),
    });

    try {
        // Use your actual backend URL
        const response = await fetch(`http://127.0.0.1:8000/compare?${params.toString()}`);
        const data = await response.json();

        if (!response.ok) {
          // Use the error message from the API
          throw new Error(data.error || "An unknown API error occurred.");
        }
        
        // The API returns { comparison_results: [...] }
        setComparisonResults(data.comparison_results);

    } catch (err) {
        setError(err.message || "Failed to fetch comparison data. Is the backend server running?");
    } finally {
        setLoading(false);
    }
  };

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
            </div>

            <div className="form-footer">
              <button className="compare-btn" onClick={handleCompare} disabled={loading || selectedDistricts.filter(d => d).length < 2}>
                  {loading ? 'Analyzing...' : 'Compare Markets'}
              </button>
            </div>
           </div>
          
           {error && <p className="error-message">{error}</p>}
           {loading && <div className="loader">Loading...</div>}

           {/* Check if there are results to display */}
           {comparisonResults && comparisonResults.length > 0 && (
            <div className="results-panel">
              <h2>Comparison Results</h2>
              <div className="comparison-table-container">
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      {/* Use the correct data structure and key */}
                      {comparisonResults.map(item => (
                        <th key={item.district}>{item.district}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>Median Price</strong><br/><span>Typical asking price</span></td>
                      {comparisonResults.map(item => <td key={item.district}>${Math.round(item.median_price).toLocaleString()}</td>)}
                    </tr>
                    <tr>
                      <td><strong>Median Price per m²</strong><br/><span>The best metric for value</span></td>
                      {comparisonResults.map(item => <td key={item.district}>${Math.round(item.median_sqm_price).toLocaleString()}/m²</td>)}
                    </tr>
                    <tr>
                      <td><strong>Typical Size</strong><br/><span>Median apartment size</span></td>
                      {comparisonResults.map(item => <td key={item.district}>{Math.round(item.median_size_m2)} m²</td>)}
                    </tr>
                     <tr>
                      <td><strong>Median Bedrooms</strong><br/><span>Typical bedroom count</span></td>
                      {comparisonResults.map(item => <td key={item.district}>{item.median_bedrooms} Beds</td>)}
                    </tr>
                    <tr>
                      <td><strong>Market Activity</strong><br/><span>Number of active listings</span></td>
                      {comparisonResults.map(item => <td key={item.district}>{item.number_of_listings} listings</td>)}
                    </tr>
                    <tr>
                      <td><strong>Price Range</strong><br/><span>10th to 90th percentile</span></td>
                      {comparisonResults.map(item => (
                        <td key={item.district}>
                            {/* Format the price range using the raw p10 and p90 numbers */}
                            ${Math.round(item.price_p10).toLocaleString()} - ${Math.round(item.price_p90).toLocaleString()}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {/* Handle the case where the API returns no results */}
          {comparisonResults && comparisonResults.length === 0 && !loading && (
             <div className="results-panel">
                <p>No listings found for the selected districts. Please try a different selection.</p>
             </div>
          )}
        </div>
    </div>
  );
};

export default MarketComparison;