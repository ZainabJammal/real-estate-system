// TimeSeriesForecasting.jsx (Updated for 5-Year Forecasting)

import { useQuery } from '@tanstack/react-query';
import { Line } from 'react-chartjs-2';
import React, { useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);



  return (
    <div className="price-estimation-container" style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>Current Property Price Estimation</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '5px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px' }}>
          {renderSelect("selectedCity", selectedCity, setSelectedCity, "City (Location)", options.cities || [], "city")}
          {renderSelect("selectedDistrict", selectedDistrict, setSelectedDistrict, "District", options.districts || [], "district")}
          {renderSelect("selectedProvince", selectedProvince, setSelectedProvince, "Province", options.provinces || [], "province")}
          {renderSelect("selectedType", selectedType, setSelectedType, "Type", options.types || [], "type")}

          <div className="form-group" style={{ marginBottom: '10px' }}>
            <label htmlFor="sizeM2" style={{ marginRight: '10px', display: 'block' }}>Size (m²): {sizeM2}</label>
            <input
              type="range"
              id="sizeM2"
              min={options.size_range?.min || 0}
              max={options.size_range?.max || 1000}
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
          disabled={isLoadingPrice}
          style={{ marginTop: '20px', padding: '10px 20px', cursor: isLoadingPrice ? 'not-allowed' : 'pointer' }}
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
};

export default CurrentPriceEstimation;