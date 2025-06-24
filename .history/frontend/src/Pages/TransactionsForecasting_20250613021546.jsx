import { useQuery } from '@tanstack/react-query';
import { Line } from 'react-chartjs-2';
import React, { useState } from "react";
import "./TransactionsForecasting.css"; // Import the new "Dashboard-Style" CSS

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
  const predefinedCities = ["Baabda, Aley, Chouf", "Beirut", "Bekaa", "Kesrouan, Jbeil", "Tripoli, Akkar"];
  const [selectedCity, setSelectedCity] = useState("");
  const [granularity, setGranularity] = useState('M');

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
    isFetching
  } = useQuery({
    queryKey: ['forecast', selectedCity],
    queryFn: () => fetchForecast(selectedCity),
    enabled: false,
    retry: false,
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (selectedCity) {
      refetch();
    } else {
      alert("Please select a city to get a forecast.");
    }
  };

  const chartData = { labels: [], datasets: [] };
  if (forecastData) {
    const { historical_data: historical, monthly_forecast: monthly, yearly_forecast: yearly } = forecastData;
    if (granularity === 'M') {
      const hLabels = historical.map(item => new Date(item.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }));
      const fLabels = monthly.map(item => new Date(item.date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' }));
      chartData.labels = [...hLabels, ...fLabels];
      chartData.datasets.push({
        label: 'Historical', data: [...historical.map(h => h.transaction_value), ...new Array(monthly.length).fill(null)],
        borderColor: 'rgb(54, 162, 235)', backgroundColor: 'rgba(54, 162, 235, 0.5)',
      });
      chartData.datasets.push({
        label: 'Forecast', data: [...new Array(historical.length).fill(null), ...monthly.map(f => f.predicted_value)],
        borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.5)', borderDash: [5, 5],
      });
    } else {
      const yearlyHist = historical.reduce((acc, item) => {
        const year = new Date(item.date).getFullYear();
        acc[year] = (acc[year] || 0) + item.transaction_value;
        return acc;
      }, {});
      const allYrs = Array.from(new Set([...Object.keys(yearlyHist).map(Number), ...yearly.map(y => y.year)])).sort();
      chartData.labels = allYrs;
      chartData.datasets.push({
        label: 'Historical Total', data: allYrs.map(y => yearlyHist[y] || null),
        borderColor: 'rgb(54, 162, 235)', backgroundColor: 'rgba(54, 162, 235, 0.5)',
      });
      chartData.datasets.push({
        label: 'Forecast Total', data: allYrs.map(y => { const f = yearly.find(i => i.year === y); return f ? f.total_value : null; }),
        borderColor: 'rgb(255, 99, 132)', backgroundColor: 'rgba(255, 99, 132, 0.5)', borderDash: [5, 5],
      });
    }
  }

  const chartOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { position: 'top' }, title: { display: true, text: `Transaction Value Forecast for ${selectedCity}` } },
    scales: { y: { beginAtZero: false } },
    interaction: { intersect: false, mode: 'index' },
  };

  return (
    <div className="forecasting-layout">
      <div className="forecasting-content">
        <div className="forecasting-title" style={{ fontSize: '20px !important' }}>
          <h1>Transaction Value Forecasting</h1>
        </div>
        
        <div className="dashboard-components">
          <div className="form-card">
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="selectedCity">City</label>
                <select id="selectedCity" value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)}>
                  <option  value=""> Select a City</option>
                  {predefinedCities.map((city) => <option key={city} value={city}>{city}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="granularity">View By</label>
                <select id="granularity" value={granularity} onChange={(e) => setGranularity(e.target.value)}>
                  <option value="M">Monthly</option>
                  <option value="Y">Yearly</option>
                </select>
              </div>
              <div className="button-group">
              <button type="submit" disabled={isLoading || isFetching || !selectedCity}>
                {isLoading || isFetching ? 'Loading...' : 'Get Forecast'}
              </button>
              </div>
            </form>
          </div>

          {isLoading || isFetching && <p className="status-message">Loading forecast...</p>}
          {error && <p className="error-message">Error: {error.message}</p>}
          
          {forecastData && !isLoading && !isFetching && (
            <div className="chart-card">
              <div className="chart-wrapper">
                <Line options={chartOptions} data={chartData} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TransactionsForecasting;