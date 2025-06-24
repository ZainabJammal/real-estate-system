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

const TransactionsForecasting = () => {
  // --- STATE MANAGEMENT ---
  const predefinedCities = ["Baabda, Aley, Chouf", "Beirut", "Bekaa", "Kesrouan, Jbeil", "Tripoli, Akkar"];
  const [selectedCity, setSelectedCity] = useState(predefinedCities[0]);
  const [granularity, setGranularity] = useState('M');

  // --- DATA FETCHING ---
  const fetchForecast = async (city) => {
    // IMPORTANT: Make sure this URL is correct for your setup
    const response = await fetch(`http://127.0.0.1:8000/forecast/xgboost/${city}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    return response.json();
  };

  const {
    data: forecastData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['forecast', selectedCity],
    queryFn: () => fetchForecast(selectedCity),
    enabled: false, 
    retry: false,
  });

  // --- EVENT HANDLERS ---
  const handleSubmit = (e) => {
    e.preventDefault();
    if (selectedCity) {
      refetch();
    } else {
      alert("Please select a city to get a forecast.");
    }
  };

  // --- CHART DATA PREPARATION (THIS SECTION IS FIXED) ---
  const chartData = {
    labels: [],
    datasets: [],
  };

  if (forecastData) {
    const historical = forecastData.historical_data || [];
    const monthlyForecast = forecastData.monthly_forecast || [];
    const yearlyForecast = forecastData.yearly_forecast || [];

    // FIX #1: Logic is now clearly separated for Monthly vs. Yearly view
    if (granularity === 'M') {
      // --- Monthly View ---
      const historicalLabels = historical.map(item => new Date(item.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }));
      const forecastLabels = monthlyForecast.map(item => new Date(item.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }));
      
      chartData.labels = [...historicalLabels, ...forecastLabels];

      chartData.datasets.push({
        label: 'Historical Data',
        data: [...historical.map(item => item.transaction_value), ...new Array(monthlyForecast.length).fill(null)],
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
      });

      chartData.datasets.push({
        label: 'Forecasted Data',
        data: [...new Array(historical.length).fill(null), ...monthlyForecast.map(item => item.predicted_value)],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderDash: [5, 5],
      });
    } else {
      // --- Yearly View (FIX #2: Only show the forecast) ---
      chartData.labels = yearlyForecast.map(item => item.year);
      
      chartData.datasets.push({
        label: 'Yearly Forecast',
        data: yearlyForecast.map(item => item.total_value),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      });
    }
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
          {isLoading ? 'Loading...' : 'Get Forecast'}
        </button>
      </form>
      
      {isLoading && <p>Loading forecast...</p>}
      {error && <p style={{ color: 'red' }}>Error: {error.message}</p>}
      
      {/* Only render the chart container if we have data to show */}
      {forecastData && !isLoading && chartData.datasets.length > 0 && (
        <div style={{ border: '1px solid #ccc', borderRadius: '8px', padding: '20px' }}>
          <div style={{ position: 'relative', height: '400px' }}>
            <Line options={chartOptions} data={chartData} />
          </div>
        </div>
      )}
    </div>
  );
};

export default TransactionsForecasting;