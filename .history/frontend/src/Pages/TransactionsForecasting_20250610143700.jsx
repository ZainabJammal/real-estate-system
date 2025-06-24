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

const TimeSeriesForecasting = () => {
  const predefinedCities = ["", "Baabda, Aley, Chouf", "Beirut", "Bekaa", "Kesrouan, Jbeil", "Tripoli, Akkar"];
  const [selectedCity, setSelectedCity] = useState(predefinedCities[0]);
  const [granularity, setGranularity] = useState('M');

  const fetchTransactionPredictions = async (cityForApi, granularityForApi) => {
    const params = {
      city_name: cityForApi === "" ? null : cityForApi,
      granularity: granularityForApi,
    };
    console.log("Attempting to fetch Transaction LSTM predictions with params:", params);
    const response = await fetch('/api/predict_transaction_timeseries', {
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

  const {
    data: predictionChartData,
    isLoading: isLoadingPredictions,
    error: predictionError,
    refetch: refetchPredictions,
    isFetching: isFetchingPredictions
  } = useQuery({
    queryKey: ['transaction_lstm_predictions', selectedCity, granularity],
    queryFn: () => {
      if (!selectedCity && selectedCity !== "") return;
      return fetchTransactionPredictions(selectedCity, granularity);
    },
    enabled: false,
    retry: false,
    onSuccess: (data) => console.log("✅ Forecast received (frontend):", data),
    onError: (err) => console.error("Error fetching Transaction LSTM prediction data:", err.message)
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    refetchPredictions();
  };

  let chartData = { labels: [], datasets: [] };
  if (predictionChartData?.historical && predictionChartData?.forecast) {
    chartData = {
      labels: [
        ...predictionChartData.historical.map(item => item.ds),
        ...predictionChartData.forecast.map(item => item.ds),
      ],
      datasets: [
        {
          label: 'Historical Transaction Value',
          data: predictionChartData.historical.map(item => item.y),
          borderColor: 'rgb(53, 162, 235)',
          backgroundColor: 'rgba(53, 162, 235, 0.5)',
          tension: 0.1
        },
        {
          label: 'Predicted Transaction Value (5-Year Forecast)',
          data: [
            ...Array(predictionChartData.historical.length).fill(null),
            ...predictionChartData.forecast.map(item => item.y)
          ],
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
          tension: 0.1
        },
      ],
    };
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: {
        display: true,
        text: 'Historical and 5-Year Forecasted Transaction Values',
        font: { size: 18 }
      },
      legend: {
        display: true,
        position: 'top'
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Date'
        },
        ticks: {
          maxRotation: 45,
          minRotation: 30,
          autoSkip: true,
          maxTicksLimit: 25
        }
      },
      y: {
        title: {
          display: true,
          text: 'Transaction Value'
        },
        beginAtZero: false
      }
    }
  };

  return (
    <div className="prediction-container" style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2>Transaction Value Time Series Forecasting</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: '20px', flexDirection:'row',  display: 'flex', flexWrap: 'nowrap', gap: '15px', justifyContent:'space-evenly',alignContent:'center',alignItems: 'center', padding: '10px', border: '1px solid #ccc', borderRadius: '5px' }}>
        <div className="form-group" style={}>
          <label htmlFor="selectedCity" style={{ marginRight: '5px' }}>City:</label>
          <select
            id="selectedCity"
            value={selectedCity}
            onChange={(e) => setSelectedCity(e.target.value)}
          >
            {predefinedCities.map((city) => (
              <option key={city || 'all-cities-option'} value={city}>
                {city || 'All Cities'}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="granularity" style={{ marginRight: '5px' }}>Time Unit:</label>
          <select id="granularity" value={granularity} onChange={(e) => setGranularity(e.target.value)}>
            <option value="M">Monthly</option>
            <option value="Y">Yearly</option>
          </select>
        </div>

        <button 
          type="submit" 
          disabled={isLoadingPredictions || isFetchingPredictions} 
          style={{ padding: '8px 15px', cursor: (isLoadingPredictions || isFetchingPredictions) ? 'not-allowed' : 'pointer' }}
        >
          {isLoadingPredictions || isFetchingPredictions ? 'Predicting...' : 'Get Forecast'}
        </button>
      </form>

      {(isLoadingPredictions || isFetchingPredictions) && <p>Loading predictions...</p>}
      {predictionError && <p className="error" style={{ color: 'red' }}>Error fetching predictions: {predictionError.message}</p>}

      {predictionChartData?.historical && predictionChartData?.forecast && (
        <div className="results" style={{ marginTop: '20px' }}>
          <p>Displaying chart for: City: <strong>{selectedCity || 'All'}</strong>, Granularity: <strong>{granularity === 'M' ? 'Monthly' : granularity === 'Y' ? 'Yearly' : granularity}</strong></p>
          <div className="chart-container" style={{ position: 'relative', height: '600px', width: '100%', margin: 'auto' }}>
            <Line options={chartOptions} data={chartData} />
          </div>
        </div>
      )}

      {!predictionChartData && !isLoadingPredictions && !isFetchingPredictions && !predictionError && (
        <p style={{ marginTop: '20px' }}>Please select parameters and click "Get Forecast" to view the time series prediction.</p>
      )}
    </div>
  );
};

export default TimeSeriesForecasting;
