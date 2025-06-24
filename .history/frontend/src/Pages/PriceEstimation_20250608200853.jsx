
import { useQuery } from '@tanstack/react-query';
// import { Line } from 'react-chartjs-2'; // Not used in this version
import React, {useEffect, useState } from "react";


// Helper to fetch data for filters
const fetchPropertyInputOptions = async () => {
  console.log("Fetching property input options from /api/property_input_options");
  const response = await fetch('/api/property_input_options'); 
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response from options API' }));
    throw new Error(errorData.error || `HTTP error! status: ${response.status} while fetching options`);
  }
  return response.json();
};

// Helper to fetch price estimation
const fetchPriceEstimation = async (params) => {
  console.log("Attempting to fetch Current Price Estimation with params:", params);
  const response = await fetch('/api/estimate_property_price', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response from estimation API' }));
    throw new Error(errorData.error || `HTTP error! status: ${response.status} while estimating price`);
  }
  return response.json();
};

const CurrentPriceEstimation = () => {
 
  const [selectedDistrict, setSelectedDistrict] = useState("");
  const [selectedType, setSelectedType] = useState("");
  const [sizeM2, setSizeM2] = useState(100);  // Default size
  const [selectedBedrooms, setSelectedBedrooms] = useState("");
  const [selectedBathrooms, setSelectedBathrooms] = useState("");

  const [estimatedPrice, setEstimatedPrice] = useState(null);
  const [isLoadingPrice, setIsLoadingPrice] = useState(false);
  const [priceError, setPriceError] = useState(null);

  const { data: options, isLoading: isLoadingOptions, error: optionsError } = useQuery({ // UNCOMMENTED
    queryKey: ['propertyInputOptions'],
    queryFn: fetchPropertyInputOptions,
    onSuccess: (data) => console.log("✅ Property Input Options received:", data),
    onError: (err) => console.error("❌ Error fetching Property Input Options:", err.message),
  });

  useEffect(() => { // UNCOMMENTED
    if (options) {
      if (options.size_range?.min !== undefined && sizeM2 < options.size_range.min) { // Initialize size if current default is too low
         setSizeM2(options.size_range.min);
      } else if (options.size_range?.min !== undefined && sizeM2 === 100 && options.size_range.min > 100) { // Or if default 100 is not sensible
         setSizeM2(options.size_range.min);
      }


     
      if (!selectedDistrict && options.districts?.length > 0) setSelectedDistrict(options.districts[0]);
      if (!selectedType && options.types?.length > 0) setSelectedType(options.types[0]);
      
      // For bedrooms/bathrooms, ensure the default is a string if options are numbers
      if (!selectedBedrooms && options.bedroom_options?.length > 0) setSelectedBedrooms(String(options.bedroom_options[0]));
      if (!selectedBathrooms && options.bathroom_options?.length > 0) setSelectedBathrooms(String(options.bathroom_options[0]));
    }
  }, [options,  selectedDistrict,  selectedType, selectedBedrooms, selectedBathrooms, sizeM2]);


  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoadingPrice(true);
    setPriceError(null);
    setEstimatedPrice(null);

    const params = {

      District: selectedDistrict,
      Type: selectedType,
      Size_m2: parseFloat(sizeM2),
      Num_Bedrooms: selectedType === "Land" ? 0 : (selectedBedrooms ? parseInt(selectedBedrooms, 10) : 0),
      Num_Bathrooms: selectedType === "Land" ? 0 : (selectedBathrooms ? parseInt(selectedBathrooms, 10) : 0),
    };

    try {
      const result = await fetchPriceEstimation(params);
      setEstimatedPrice(result.prediction);
    } catch (error) {
      setPriceError(error.message);
    } finally {
      setIsLoadingPrice(false);
    }
  };

  if (isLoadingOptions) return <p>Loading filter options...</p>;
  if (optionsError) return <p className="error" style={{ color: 'red' }}>Error loading filter options: {optionsError.message}</p>;
  // It's possible options is fetched but is null/empty if the API returns that for some reason
  if (!options || Object.keys(options).length === 0) return <p>No filter options available or failed to load. Check API response.</p>;


  const renderSelect = (id, value, onChange, label, selectOptions, keyPrefix = "") => (
    <div className="form-group" style={{ marginBottom: '10px' }}>
      <label htmlFor={id} style={{ marginRight: '10px', display: 'block' }}>{label}:</label>
      <select id={id} value={value} onChange={e => onChange(e.target.value)} style={{ padding: '8px', width: '200px' }} disabled={!selectOptions || selectOptions.length === 0}>
        {(!selectOptions || selectOptions.length === 0) && <option value="">Loading or N/A</option>}
        {selectOptions && selectOptions.map((opt) => (
          <option key={`${keyPrefix}-${opt}`} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );

  return (
    <div className="price-estimation-container" style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>Current Property Price Estimation</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '5px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px' }}>
          {renderSelect("selectedDistrict", selectedDistrict, setSelectedDistrict, "District", options.districts || [], "district")}
          {renderSelect("selectedType", selectedType, setSelectedType, "Type", options.types || [], "type")}

          <div className="form-group" style={{ marginBottom: '10px' }}>
            <label htmlFor="sizeM2" style={{ marginRight: '10px', display: 'block' }}>Size (m²): {sizeM2}</label>
            <input
              type="range"
              id="sizeM2"
              min={options.size_range?.min ?? 0} // Use ?? for default if min/max are undefined
              max={options.size_range?.max ?? 1000}
              value={sizeM2}
              onChange={(e) => setSizeM2(e.target.value)}
              style={{ width: '200px' }}
            />
          </div>

          {selectedType !== "Land" && (
            <>
              {renderSelect("selectedBedrooms", selectedBedrooms, setSelectedBedrooms, "Bedrooms", options.bedroom_options?.map(String) || [], "bed")}
              {renderSelect("selectedBathrooms", selectedBathrooms, setSelectedBathrooms, "Bathrooms", options.bathroom_options?.map(String) || [], "bath")}
            </>
          )}
        </div>

        <button
          type="submit"
          disabled={isLoadingPrice  || !selectedDistrict || !selectedType } // Basic validation
          style={{ marginTop: '20px', padding: '10px 20px', cursor: (isLoadingPrice || !selectedDistrict || !selectedType) ? 'not-allowed' : 'pointer' }}
        >
          {isLoadingPrice ? 'Estimating...' : 'Estimate Price'}
        </button>
      </form>

      {isLoadingPrice && <p>Loading estimated price...</p>}
      {priceError && <p className="error" style={{ color: 'red' }}>Error estimating price: {priceError}</p>}
      
      {estimatedPrice !== null && (
        <div className="results" style={{ marginTop: '20px', padding: '15px', border: '1px solid #007bff', borderRadius: '5px', backgroundColor: '#e7f3ff' }}>
          <h3>Estimated Property Price:</h3>
          <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#0056b3' }}>
            ${estimatedPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
        </div>
      )}
    </div>
  );
}; // CORRECTED CLOSING

export default CurrentPriceEstimation;

