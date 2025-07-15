import { useState, useEffect } from 'react';
import axios from 'axios'; 
import Autosuggest from 'react-autosuggest'; 
import './MarketComparison.css';

// Your API endpoint. For offline, this points to your local server.
const API_URL = "http://127.0.0.1:8000"; 

// The icon for the 'add' button (replace with your actual icon component if you have one)
const PlusIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z" />
  </svg>
);

const MarketComparison = () => {
  // State to manage the list of selected districts
  const [selectedDistricts, setSelectedDistricts] = useState(['', '']); // Start with two empty selectors
  
  // State for other filters
  const [propertyType, setPropertyType] = useState('Apartment');
  const [bedrooms, setBedrooms] = useState([2, 3]); // Example, can be made dynamic

  // State to hold data for the dropdowns
  const [allDistricts, setAllDistricts] = useState([]);

  // State for API interaction and results
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [comparisonData, setComparisonData] = useState(null);

  // --- Fetch the list of all districts on component mount ---
  useEffect(() => {
    // In a real app, this would fetch from your /config/filters endpoint
    // For this offline example, we'll hardcode them based on your CSV.
    const districtsFromData = [
      "Beirut", "El Metn", "Kesrouane", "Jbeil", "Baabda", "Aley",
      "Batroun"
    ].sort();
    setAllDistricts(districtsFromData);
  }, []);

  // --- Event Handlers ---
  const handleDistrictChange = (index, value) => {
    const newSelectedDistricts = [...selectedDistricts];
    newSelectedDistricts[index] = value;
    setSelectedDistricts(newSelectedDistricts);
  };

  const handleAddDistrict = () => {
    // Limit the number of comparisons to keep the table readable
    if (selectedDistricts.length < 5) {
      setSelectedDistricts([...selectedDistricts, '']);
    }
  };
  
  const handleRemoveDistrict = (index) => {
    // Prevent removing below two inputs
    if (selectedDistricts.length > 2) {
      const newSelectedDistricts = selectedDistricts.filter((_, i) => i !== index);
      setSelectedDistricts(newSelectedDistricts);
    }
  };

  const handleCompare = async () => {
    // Filter out any empty selections and duplicates
    const validDistricts = [...new Set(selectedDistricts.filter(d => d))];

    if (validDistricts.length < 2) {
      setError("Please select at least two different districts to compare.");
      return;
    }

    setLoading(true);
    setError('');
    setComparisonData(null);

    // Construct the query parameters
    const params = new URLSearchParams({
        type: propertyType,
        districts: validDistricts.join(','),
        // bedrooms: bedrooms.join(',') // Example if your API supports it
    });

    try {
        // This is where you would call your real API
        // For now, we simulate a response
        // const response = await fetch(`${API_URL}/api/v1/compare?${params.toString()}`);
        console.log(`Simulating API call: GET /api/v1/compare?${params.toString()}`);
        
        // --- SIMULATED API RESPONSE ---
        await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate network delay
        const mockData = {
            "comparison_results": validDistricts.map(d => ({
                "district": d,
                "median_price": 200000 + Math.random() * 500000,
                "median_sqm_price": 1000 + Math.random() * 2000,
                "typical_size": 150 + Math.random() * 100,
                "market_activity": 100 + Math.floor(Math.random() * 800),
                "price_p10": 100000 + Math.random() * 50000,
                "price_p90": 800000 + Math.random() * 1000000,
            }))
        };
        setComparisonData(mockData);

    } catch (err) {
        setError("Failed to fetch comparison data. Please try again.");
    } finally {
        setLoading(false);
    }
  };
  
  return (
     <div className="comparison-layout">
       <div className="comparison-content">
          <div className="estimator-title" style={{ fontSize: '20px' }}>
            <h1>The Market Comparison Tool</h1>
          </div>
        <div className="dashboard-components">
           <div className="form-card">
          <div className="form-card">
             {error && <p className="error-message">{error}</p>}
           
              <div className="form-group">
                <div className="district-selectors">
                  <label>Districts to Compare</label>
          {selectedDistricts.map((district, index) => (
            <div key={index} className="district-input-row">
              <select value={district} onChange={(e) => handleDistrictChange(index, e.target.value)}>
                <option value="">-- Select District --</option>
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

        <div className="filter-selectors">
            <div className="form-group">
                <label>Property Type</label>
                <select value={propertyType} onChange={e => setPropertyType(e.target.value)}>
                    <option value="Apartment">Apartment</option>
                    {/* <option value="House/Villa">House/Villa</option>
                    <option value="Land">Land</option> */}
                    {/* Add other types as needed */}
                </select>
            </div>
        </div>
      <div className="form-group" style={{ marginLeft: '50%' }}></div>
        <button className="compare-btn" onClick={handleCompare} disabled={loading || selectedDistricts.filter(d => d).length < 2}>
            {loading ? 'Analyzing...' : 'Compare'}
        </button>
       
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
                  <td><strong>Typical Size</strong><br/><span>Average for selected type</span></td>
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
    </div>
    </div>
    </div>
  );
};

export default MarketComparison;