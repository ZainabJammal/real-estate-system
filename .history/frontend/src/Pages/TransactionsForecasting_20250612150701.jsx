// TransactionsForecasting.jsx (Corrected)

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
  // const [periods, setPeriods] = useState(12); // Add state for number of periods to forecast

  const fetchTransactionPredictions = async (cityForApi, granularityForApi, periodsForApi) => {
    // FIX: Send 'periods' in the request body
    const params = {
      city: cityForApi,
      granularity: granularityForApi,
      periods: periodsForApi,
    };
    console.log("Attempting to fetch Transaction LSTM predictions with params:", params);
    const response = await fetch('/api/forecast', {
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
    data: predictionData, // Renamed for clarity
    isLoading,
    error,
    refetch,
    isFetching
  } = useQuery({
    // FIX: Include periods in the queryKey for proper caching
    queryKey: ['transaction_lstm_predictions', selectedCity, granularity], 
    queryFn: () => {
      // FIX: Ensure a city is selected before fetching
      if (!selectedCity) return Promise.resolve(null);
      return fetchTransactionPredictions(selectedCity, granularity);
    },
    enabled: false,
    retry: false,
    onSuccess: (data) => console.log("✅ Forecast received (frontend):", data),
    onError: (err) => console.error("Error fetching Transaction LSTM prediction data:", err.message)
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    // FIX: Add a check to prevent submitting with no city selected
    if (!selectedCity) {
      alert("Please select a city to get a forecast.");
      return;
    }
    refetch();
  };

  // --- MAJOR FIX: Process the data structure the API actually sends ---
  let chartData = { labels: [], datasets: [] };
  // The API response has a 'forecast' key.
  if (predictionData?.forecast) {
    chartData = {
      // The forecast objects have a 'date' key
      labels: predictionData.forecast.map(item => item.date),
      datasets: [
        {
          label: `Forecasted Transaction Value for ${selectedCity}`,
          // The forecast objects have a 'predicted_value' key
          data: predictionData.forecast.map(item => item.predicted_value),
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
          tension: 0.1
        },
      ],
    };
  }

  const chartOptions = { /* ... your chart options are fine, no changes needed ... */ };

  return (
    <div className="prediction-container" style={{ alignItems: 'center', padding: '20px', width: '100%', borderRadius: '5px' }}>
      <div><h1 style={{ fontFamily:'Cantarell, "Open Sans", "Helvetica Neue", sans-serif',paddingTop:'20px',  fontSize: '24px', marginBottom: '20px' }}>Transaction Value Time Series Forecasting</h1>

      </div>
      <section style={{ height: '150px', overflow: 'hidden', flexDirection: 'row',  }}>
        <form onSubmit={handleSubmit} style={{ paddingTop:'35px',flex: '1', display: 'flex', borderRadius:'5px', border: '1px solid var(--background-light-gray-f)',
         width:'100%', height:'400px', fontSize:'20px'}}>
        <div className="form-group">
          <label htmlFor="selectedCity" style={{ paddingLeft: '10px', fontSize: '20px', 
            fontFamily:'Cantarell, "Open Sans", "Helvetica Neue", sans-serif' }}>City:</label>
         
          <select style={{ backgroundColor:'#2C3E50',padding: '8px', fontWeight: '400', color:'white',
              marginLeft: '20px',
              height: '35px',
              width: '150px' }}
          id="selectedCity" value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)}>
            <option value="">-- Select a City --</option>
            {predefinedCities.slice(1).map((city) => (
              <option key={city} value={city}>{city}</option>
            ))}
          </select>
          </div>
 

        <div className="form-group" style={{paddingLeft: '90px'}} >
          <label htmlFor="granularity">Time Unit:</label>
          <select style={{ backgroundColor:'#2C3E50',padding: '8px', fontWeight: '400', color:'white',
              marginLeft: '20px',
              height: '35px',
              width: '150px' }} id="granularity" value={granularity} onChange={(e) => setGranularity(e.target.value)}>
            <option value="M">Monthly</option>
            <option value="Y">Yearly</option>
          </select>
        </div>

        <div style={{ paddingLeft: '50px' }}><button 
          type="submit" 
          // FIX: Also disable the button if no city is selected
          disabled={isLoading || isFetching || !selectedCity} 
          style={{ flexDirection:'row',padding: '8px 15px', cursor: (isLoading || isFetching || !selectedCity) ? 'not-allowed' : 'pointer' }}
        >
          {isFetching ? 'Predicting...' : 'Get Forecast'}
        </button></div>
      </form></section>

      {isFetching && <p>Loading predictions...</p>}
      {error && <p className="error" style={{ color: 'red' }}>Error fetching predictions: {error.message}</p>}

      {/* FIX: Update the condition to check for the correct data structure */}
      {predictionData?.forecast && !isFetching && (
        <div className="results" style={{  borderRadius:'5px', border: '1px solid var(--background-light-gray-f)', backgroundColor: 'var(--background-light-white-p)',
          boxShadow: '0px 1px 5px 1px var(--background-light-gray-p)',  transition: 'transform 0.225s ease-in-out' }}>
          <p>Displaying forecast for: City: <strong>{selectedCity}</strong>, Granularity: <strong>{granularity === 'M' ? 'Monthly' : 'Yearly'}</strong></p>
          <div className="chart-container" style={{ width: '100%', height: '350px', margin: 'auto' }}>
            <Line options={chartOptions} data={chartData} />
          </div>
        </div>
      )}

      {!predictionData?.forecast && !isFetching && !error && (
        <p style={{ marginTop: '20px' }}>Please select parameters and click "Get Forecast" to view the time series prediction.</p>
      )}
      
    </div>
    
  );
};


export default TimeSeriesForecasting;