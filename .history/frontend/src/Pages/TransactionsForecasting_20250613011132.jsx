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

// This is the main component for our forecasting tool
const TransactionsForecasting = () => {
  // --- STATE MANAGEMENT ---
  // List of cities available for forecasting
  const predefinedCities = ["Baabda, Aley, Chouf", "Beirut", "Bekaa", "Kesrouan, Jbeil", "Tripoli, Akkar"];
  // State to hold the user's selected city
  const [selectedCity, setSelectedCity] = useState(predefinedCities[0]);
  // State to control the chart's display granularity (Monthly vs Yearly)
  const [granularity, setGranularity] = useState('M');

  // --- DATA FETCHING ---
  // A single function to call our backend API
  const fetchForecast = async (city) => {
    // The API endpoint uses a GET request with the city name in the URL
    cconst response = await fetch(`http://127.0.0.1:8000/api/forecast/xgboost/${city}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    return response.json();
  };

  // React Query hook to manage the API call
  const {
    data: forecastData, // This will hold the complete response: { historical, monthly, yearly }
    isLoading,
    error,
    refetch, // We call this function to trigger the API call
  } = useQuery({
    queryKey: ['forecast', selectedCity], // A unique key for this query
    queryFn: () => fetchForecast(selectedCity),
    // `enabled: false` means the query will not run automatically. It waits for `refetch()`.
    enabled: false, 
    retry: false, // Don't retry on failure
    onSuccess: (data) => console.log("✅ Forecast received:", data),
    onError: (err) => console.error("❌ Error fetching forecast:", err.message),
  });

  // --- EVENT HANDLERS ---
  const handleSubmit = (e) => {
    e.preventDefault();
    if (selectedCity) {
      // Trigger the API call
      refetch();
    } else {
      alert("Please select a city to get a forecast.");
    }
  };

  // --- CHART DATA PREPARATION ---
  // This logic runs whenever `forecastData` or `granularity` changes.
  const chartData = {
    labels: [],
    datasets: [],
  };

  if (forecastData) {
    const historical = forecastData.historical_data || [];
    const monthly = forecastData.monthly_forecast || [];
    const yearly = forecastData.yearly_forecast || [];

    // Determine which data to show based on granularity
    let displayForecast, labels;
    if (granularity === 'M') {
      labels = monthly.map(item => new Date(item.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }));
      displayForecast = monthly.map(item => item.predicted_value);
    } else { // Yearly
      labels = yearly.map(item => item.year);
      displayForecast = yearly.map(item => item.total_value);
    }

    // Combine historical and forecast labels
    const historicalLabels = historical.map(item => new Date(item.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }));
    
    // For yearly view, we just show the forecast
    chartData.labels = granularity === 'Y' ? labels : [...historicalLabels, ...labels];

    // Prepare datasets for the chart
    const historicalDataset = {
      label: 'Historical Data',
      data: [...historical.map(item => item.transaction_value), ...new Array(displayForecast.length).fill(null)],
      borderColor: 'rgb(54, 162, 235)',
      backgroundColor: 'rgba(54, 162, 235, 0.5)',
      tension: 0.1,
    };

    const forecastDataset = {
      label: 'Forecasted Data',
      data: [...new Array(historical.length).fill(null), ...displayForecast],
      borderColor: 'rgb(255, 99, 132)',
      backgroundColor: 'rgba(255, 99, 132, 0.5)',
      borderDash: [5, 5], // Dashed line for predictions
      tension: 0.1,
    };
    
    // In yearly view, we don't need the historical line
    chartData.datasets = granularity === 'Y' ? [forecastDataset] : [historicalDataset, forecastDataset];
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: `Transaction Value Forecast for ${selectedCity}` },
    },
    scales: { y: { beginAtZero: false } },
    interaction: { intersect: false, mode: 'index' },
  };

  // --- JSX RENDER ---
  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>Transaction Value Time Series Forecasting</h1>
      
      {/* --- FORM --- */}
      <form onSubmit={handleSubmit} style={{ display: 'flex', alignItems: 'center', gap: '20px', marginBottom: '20px', padding: '15px', border: '1px solid #ccc', borderRadius: '8px' }}>
        <div>
          <label htmlFor="selectedCity" style={{ marginRight: '10px' }}>City:</label>
          <select 
            id="selectedCity" 
            value={selectedCity} 
            onChange={(e) => setSelectedCity(e.target.value)}
            style={{ padding: '8px' }}
          >
            {predefinedCities.map((city) => (
              <option key={city} value={city}>{city}</option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="granularity" style={{ marginRight: '10px' }}>View By:</label>
          <select 
            id="granularity" 
            value={granularity} 
            onChange={(e) => setGranularity(e.target.value)}
            style={{ padding: '8px' }}
          >
            <option value="M">Monthly</option>
            <option value="Y">Yearly</option>
          </select>
        </div>

        <button type="submit" disabled={isLoading || !selectedCity} style={{ padding: '8px 15px', cursor: 'pointer' }}>
          {isLoading ? 'Predicting...' : 'Get Forecast'}
        </button>
      </form>

      {/* --- RESULTS & CHART --- */}
      {isLoading && <p>Loading forecast...</p>}
      {error && <p style={{ color: 'red' }}>Error: {error.message}</p>}
      
      {forecastData && !isLoading && (
        <div style={{ border: '1px solid #ccc', borderRadius: '8px', padding: '20px' }}>
          <p>
            Displaying data for <strong>{selectedCity}</strong> | 
            Viewing by <strong>{granularity === 'M' ? 'Month' : 'Year'}</strong>
          </p>
          <div style={{ position: 'relative', height: '400px' }}>
            <Line options={chartOptions} data={chartData} />
          </div>
        </div>
      )}
      
      {!forecastData && !isLoading && <p>Please select a city and click "Get Forecast" to see the results.</p>}
    </div>
  );
};

export default TransactionsForecasting;