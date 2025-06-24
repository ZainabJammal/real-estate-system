import { useQuery } from '@tanstack/react-query';
import { Line } from 'react-chartjs-2';
import React, { useState } from "react";
import "./TransactionsForecasting.css"; // Import the new CSS file

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
  const [selectedCity, setSelectedCity] = useState(""); // Default to empty
  const [granularity, setGranularity] = useState('M');

  // --- DATA FETCHING ---
  const fetchForecast = async (city) => {
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
    isFetching,
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

  // --- CHART DATA PREPARATION ---
  const chartData = {
    labels: [],
    datasets: [],
  };

  if (forecastData) {
    const historical = forecastData.historical_data || [];
    const monthlyForecast = forecastData.monthly_forecast || [];
    const yearlyForecast = forecastData.yearly_forecast || [];

    if (granularity === 'M') {
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
      const yearlyHistoricalSums = historical.reduce((acc, item) => {
        const year = new Date(item.date).getFullYear();
        acc[year] = (acc[year] || 0) + item.transaction_value;
        return acc;
      }, {});
      const historicalYears = Object.keys(yearlyHistoricalSums).map(Number);
      const forecastYears = yearlyForecast.map(item => item.year);
      const allYears = Array.from(new Set([...historicalYears, ...forecastYears])).sort();
      chartData.labels = allYears;
      chartData.datasets.push({
        label: 'Historical Yearly Total',
        data: allYears.map(year => yearlyHistoricalSums[year] || null),
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
      });
      chartData.datasets.push({
        label: 'Forecasted Yearly Total',
        data: allYears.map(year => {
          const forecastItem = yearlyForecast.find(item => item.year === year);
          return forecastItem ? forecastItem.total_value : null;
        }),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderDash: [5, 5],
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
    <div className="forecasting-container">
      <h1 className="forecasting-header">Transaction Value Time Series Forecasting</h1>
      
      <form onSubmit={handleSubmit} className="forecasting-form">
        <div className="form-group">
          <label htmlFor="selectedCity">City:</label>
          <select  
            id="selectedCity" 
            value={selectedCity} 
            onChange={(e) => setSelectedCity(e.target.value)}
          >
            <option value="">-- Select a City --</option>
            {predefinedCities.map((city) => (
              <option key={city} value={city}>{city}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="granularity">View By:</label>
          <select 
            id="granularity" 
            value={granularity} 
            onChange={(e) => setGranularity(e.target.value)}
          >
            <option value="M">Monthly</option>
            <option value="Y">Yearly</option>
          </select>
        </div>

        <button type="submit" disabled={isLoading || isFetching || !selectedCity} className="forecast-button">
          {isLoading || isFetching ? 'Loading...' : 'Get Forecast'}
        </button>
      </form>
      
      {isLoading || isFetching && <p className="status-message">Loading forecast...</p>}
      {error && <p className="error-message">Error: {error.message}</p>}
      
      {forecastData && !isLoading && !isFetching && (
        <div className="forecasting-results">
          <div className="chart-container">
            <Line options={chartOptions} data={chartData} />
          </div>
        </div>
      )}
    </div>
  );
};

export default TransactionsForecasting;