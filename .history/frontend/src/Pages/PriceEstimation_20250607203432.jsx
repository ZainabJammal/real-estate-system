// TimeSeriesForecasting.jsx (Updated for 5-Year Forecasting)
import { useQuery } from '@tanstack/react-query';
import { Line } from 'react-chartjs-2';
import React, {useEffect, useState } from "react";


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

  // Helper to fetch data for filters
const fetchPropertyInputOptions = async () => {
  const response = await fetch('/api/estimate_price');
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
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
    const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
  }
  return response.json();
};

const CurrentPriceEstimation = () => {
  const [selectedDistrict, setSelectedDistrict] = useState("", "Baabda", "Kesrouane", "Koura", "Hasbaya", "Aley", "Beirut", "Zahle", "Chouf", "Zgharta", "Jbeil", "Batroun", "El Metn");
  const [selectedType, setSelectedType] = useState("[
  {
    "property_type": "Restaurant"
   "Gas Station"
  },
  {
    "property_type": "Residential Building"
  },
  {
    "property_type": "Office"
  },
  {
    "property_type": "Factory"
  },
  {
    "property_type": "House/Villa"
  },
  {
    "property_type": "Chalet"
  },
  {
    "property_type": "Commercial Building"
  },
  {
    "property_type": "Warehouse"
  },
  {
    "property_type": "Apartment"
  },
  {
    "property_type": "Shop"
  },
  {
    "property_type": "Land"
  }
]");
  const [sizeM2, setSizeM2] = useState(100); // Default size
  const [selectedBedrooms, setSelectedBedrooms] = useState("");
  const [selectedBathrooms, setSelectedBathrooms] = useState("");

  const [estimatedPrice, setEstimatedPrice] = useState(null);
  const [isLoadingPrice, setIsLoadingPrice] = useState(false);
  const [priceError, setPriceError] = useState(null);

  // const { data: options, isLoading: isLoadingOptions, error: optionsError } = useQuery({
  //   queryKey: ['propertyInputOptions'],
  //   queryFn: fetchPropertyInputOptions,
  // });

  useEffect(() => {
    // Set default size based on fetched range if available
  //   if (options?.size_range?.min) {
  //     setSizeM2(options.size_range.min);
  //   }
  //   //  // Set default values for dropdowns once options are loaded
  //   // if (options?.districts?.length > 0 && !selectedDistrict) setSelectedDistrict(options.districts[0]);
  //   if (options?.types?.length > 0 && !selectedType) setSelectedType(options.types[0]);
  //   if (options?.bedroom_options?.length > 0 && !selectedBedrooms) setSelectedBedrooms(String(options.bedroom_options[0]));
  //   if (options?.bathroom_options?.length > 0 && !selectedBathrooms) setSelectedBathrooms(String(options.bathroom_options[0]));

  // }, [options,  selectedDistrict, selectedType, selectedBedrooms, selectedBathrooms]);


  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoadingPrice(true);
    setPriceError(null);
    setEstimatedPrice(null);

    const params = {

      District: selectedDistrict,
      Type: selectedType,
      Size_m2: parseFloat(sizeM2),
      Num_Bedrooms: selectedType === "Land" ? 0 : parseInt(selectedBedrooms, 10),
      Num_Bathrooms: selectedType === "Land" ? 0 : parseInt(selectedBathrooms, 10),
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
  if (!options) return <p>No filter options available.</p>;


  const renderSelect = (id, value, onChange, label, Soptions, SkeyPrefix = "") => (
    <div className="form-group" style={{ marginBottom: '10px' }}>
      <label htmlFor={id} style={{ marginRight: '10px', display: 'block' }}>{label}:</label>
      <select id={id} value={value} onChange={e => onChange(e.target.value)} style={{ padding: '8px', width: '200px' }}>
        {Soptions.map((opt) => (
          <option key={`${SkeyPrefix}-${opt}`} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );

  



  // let chartData = { labels: [], datasets: [] };
  // if (predictionChartData?.historical && predictionChartData?.forecast) {
  //   chartData = {
  //     labels: [
  //       ...predictionChartData.historical.map(item => item.ds),
  //       ...predictionChartData.forecast.map(item => item.ds),
  //     ],
  //     datasets: [
  //       {
  //         label: 'Historical Transaction Value',
  //         data: predictionChartData.historical.map(item => item.y),
  //         borderColor: 'rgb(53, 162, 235)',
  //         backgroundColor: 'rgba(53, 162, 235, 0.5)',
  //         tension: 0.1
  //       },
  //       {
  //         label: 'Predicted Transaction Value (5-Year Forecast)',
  //         data: [
  //           ...Array(predictionChartData.historical.length).fill(null),
  //           ...predictionChartData.forecast.map(item => item.y)
  //         ],
  //         borderColor: 'rgb(255, 99, 132)',
  //         backgroundColor: 'rgba(255, 99, 132, 0.5)',
  //         tension: 0.1
  //       },
  //     ],
  //   };
  // }

  // const chartOptions = {
  //   responsive: true,
  //   maintainAspectRatio: false,
  //   plugins: {
  //     title: {
  //       display: true,
  //       text: 'Historical and 5-Year Forecasted Transaction Values',
  //       font: { size: 18 }
  //     },
  //     legend: {
  //       display: true,
  //       position: 'top'
  //     },
  //     tooltip: {
  //       mode: 'index',
  //       intersect: false
  //     }
  //   },
  //   scales: {
  //     x: {
  //       title: {
  //         display: true,
  //         text: 'Date'
  //       },
  //       ticks: {
  //         maxRotation: 45,
  //         minRotation: 30,
  //         autoSkip: true,
  //         maxTicksLimit: 25
  //       }
  //     },
  //     y: {
  //       title: {
  //         display: true,
  //         text: 'Transaction Value'
  //       },
  //       beginAtZero: false
  //     }
  //   }
  // };


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
}) ; }

export default CurrentPriceEstimation;