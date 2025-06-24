--- START OF FILE PriceEstimation.jsx ---

import React, { useState, useEffect } from "react";
import { useQuery } from '@tanstack/react-query';

// Helper to fetch filter options from the API
const fetchPropertyInputOptions = async () => {
  const response = await fetch('/api/property_input_options');
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || 'Failed to fetch filter options');
  }
  return response.json();
};

// Helper to post data for price estimation
const fetchPriceEstimation = async (params) => {
  const response = await fetch('/api/estimate_property_price', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || 'Failed to get price estimation');
  }
  return response.json();
};

const CurrentPriceEstimation = () => {
  const [selectedDistrict, setSelectedDistrict] = useState("");
  const [selectedType, setSelectedType] = useState("");
  const [sizeM2, setSizeM2] = useState(100);
  const [selectedBedrooms, setSelectedBedrooms] = useState("");
  const [selectedBathrooms, setSelectedBathrooms] = useState("");

  const [estimatedPrice, setEstimatedPrice] = useState(null);
  const [isLoadingPrice, setIsLoadingPrice] = useState(false);
  const [priceError, setPriceError] = useState(null);

  const { data: options, isLoading: isLoadingOptions, error: optionsError } = useQuery({
    queryKey: ['propertyInputOptions'],
    queryFn: fetchPropertyInputOptions,
    staleTime: Infinity, // These options don't change often, cache them
  });
  
  // Set default values once options are loaded
  useEffect(() => {
    if (options) {
      if (!selectedDistrict && options.districts?.length > 0) setSelectedDistrict(options.districts[0]);
      if (!selectedType && options.types?.length > 0) setSelectedType(options.types[0]);
      if (!selectedBedrooms && options.bedroom_options?.length > 0) setSelectedBedrooms(String(options.bedroom_options[0]));
      if (!selectedBathrooms && options.bathroom_options?.length > 0) setSelectedBathrooms(String(options.bathroom_options[0]));
      if (options.size_range?.min && sizeM2 < options.size_range.min) setSizeM2(options.size_range.min);
    }
  }, [options, selectedDistrict, selectedType, selectedBedrooms, selectedBathrooms, sizeM2]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoadingPrice(true);
    setPriceError(null);
    setEstimatedPrice(null);

    // CORRECTED: Payload keys are lowercase to match the backend API
    const params = {
      district: selectedDistrict,
      type: selectedType,
      size_m2: parseFloat(sizeM2),
      bedrooms: selectedType.toLowerCase() === "land" ? 0 : parseInt(selectedBedrooms, 10),
      bathrooms: selectedType.toLowerCase() === "land" ? 0 : parseInt(selectedBathrooms, 10),
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
  if (optionsError) return <p style={{ color: 'red' }}>Error: {optionsError.message}</p>;

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>Current Property Price Estimation</h2>
      <p>Select property features to get an estimated market price.</p>
      
      <form onSubmit={handleSubmit} style={{ marginBottom: '20px', padding: '20px', border: '1px solid #ccc', borderRadius: '8px', background: '#f9f9f9' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '25px', alignItems: 'flex-end' }}>
          {/* District Selector */}
          <div>
            <label htmlFor="district" style={{ display: 'block', marginBottom: '5px' }}>District</label>
            <select id="district" value={selectedDistrict} onChange={e => setSelectedDistrict(e.target.value)} style={{ padding: '8px', width: '200px' }}>
              {options?.districts?.map(opt => <option key={opt} value={opt}>{opt}</option>)}
            </select>
          </div>
          {/* Type Selector */}
          <div>
            <label htmlFor="type" style={{ display: 'block', marginBottom: '5px' }}>Type</label>
            <select id="type" value={selectedType} onChange={e => setSelectedType(e.target.value)} style={{ padding: '8px', width: '200px' }}>
              {options?.types?.map(opt => <option key={opt} value={opt}>{opt}</option>)}
            </select>
          </div>
          {/* Size Slider */}
          <div>
            <label htmlFor="sizeM2" style={{ display: 'block', marginBottom: '5px' }}>Size (m²): <strong>{sizeM2}</strong></label>
            <input type="range" id="sizeM2" min={options?.size_range?.min ?? 0} max={options?.size_range?.max ?? 1000} value={sizeM2} onChange={e => setSizeM2(e.target.value)} style={{ width: '200px' }} />
          </div>
          {/* Conditional Bedroom/Bathroom selectors */}
          {selectedType.toLowerCase() !== "land" && (
            <>
              <div>
                <label htmlFor="bedrooms" style={{ display: 'block', marginBottom: '5px' }}>Bedrooms</label>
                <select id="bedrooms" value={selectedBedrooms} onChange={e => setSelectedBedrooms(e.target.value)} style={{ padding: '8px', width: '120px' }}>
                  {options?.bedroom_options?.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                </select>
              </div>
              <div>
                <label htmlFor="bathrooms" style={{ display: 'block', marginBottom: '5px' }}>Bathrooms</label>
                <select id="bathrooms" value={selectedBathrooms} onChange={e => setSelectedBathrooms(e.target.value)} style={{ padding: '8px', width: '120px' }}>
                  {options?.bathroom_options?.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                </select>
              </div>
            </>
          )}
        </div>
        <button type="submit" disabled={isLoadingPrice || !selectedDistrict || !selectedType} style={{ marginTop: '25px', padding: '12px 24px', cursor: 'pointer', fontSize: '16px' }}>
          {isLoadingPrice ? 'Estimating...' : 'Estimate Price'}
        </button>
      </form>

      {/* --- RESULT DISPLAY --- */}
      {isLoadingPrice && <p>Loading estimated price...</p>}
      {priceError && <p style={{ color: 'red', fontWeight: 'bold' }}>Error: {priceError}</p>}
      {estimatedPrice !== null && (
        <div style={{ marginTop: '20px', padding: '20px', border: '2px solid #007bff', borderRadius: '8px', backgroundColor: '#e7f3ff' }}>
          <h3>Estimated Property Price:</h3>
          <p style={{ fontSize: '28px', fontWeight: 'bold', color: '#0056b3', margin: '0' }}>
            ${estimatedPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
        </div>
      )}
    </div>
  );
};

export default CurrentPriceEstimation;