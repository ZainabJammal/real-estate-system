
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
  const [lstmGranularity, setLstmGranularity] = useState('Y');
  const [activeTab, setActiveTab] = useState('single'); // 'single' or 'lstm'

  // Fetch historical data for a specific city
  const fetchHistoricalData = async (cityForApi, granularityForApi) => {
    const params = {
      city: cityForApi,
      granularity: granularityForApi,
      start_year: 2012, // Last 5 years from 2016 (our training end)
      end_year: 2016
    };
    console.log("Fetching historical data with params:", params);
    
    // We'll use a mock endpoint for now - in a real implementation, 
    // you'd create an API endpoint to fetch historical data
    const response = await fetch('/api/historical_data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    
    if (!response.ok) {
      // If historical endpoint doesn't exist, return empty data
      console.log("Historical data endpoint not available, using empty data");
      return { historical: [] };
    }
    return response.json();
  };

  // Fetch historical data for all cities (for LSTM tab)
  const fetchAllHistoricalData = async (granularityForApi) => {
    const params = {
      granularity: granularityForApi,
      start_year: 2012,
      end_year: 2016
    };
    console.log("Fetching all historical data with params:", params);
    
    const response = await fetch('/api/all_historical_data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    
    if (!response.ok) {
      console.log("All historical data endpoint not available, using empty data");
      return { historical: {} };
    }
    return response.json();
  };

  // Original single-city forecasting
  const fetchTransactionPredictions = async (cityForApi, granularityForApi) => {
    const params = {
      city: cityForApi,
      granularity: granularityForApi,
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

  // New LSTM forecasting for all cities
  const fetchLSTMForecasts = async (granularityForApi) => {
    const params = {
      granularity: granularityForApi,
    };
    console.log("Attempting to fetch LSTM forecasts for all cities with params:", params);
    const response = await fetch('/api/lstm_forecast', {
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

  // Historical data query for single city
  const {
    data: historicalData,
    isLoading: historicalIsLoading,
    error: historicalError,
  } = useQuery({
    queryKey: ['historical_data', selectedCity, granularity], 
    queryFn: () => {
      if (!selectedCity) return Promise.resolve({ historical: [] });
      return fetchHistoricalData(selectedCity, granularity);
    },
    enabled: !!selectedCity,
    retry: false,
  });

  // Single city forecast query
  const {
    data: predictionData,
    isLoading,
    error,
    refetch,
    isFetching
  } = useQuery({
    queryKey: ['transaction_lstm_predictions', selectedCity, granularity], 
    queryFn: () => {
      if (!selectedCity) return Promise.resolve(null);
      return fetchTransactionPredictions(selectedCity, granularity);
    },
    enabled: false,
    retry: false,
    onSuccess: (data) => console.log("✅ Forecast received (frontend):", data),
    onError: (err) => console.error("Error fetching Transaction LSTM prediction data:", err.message)
  });

  // Historical data query for all cities (LSTM tab)
  const {
    data: allHistoricalData,
    isLoading: allHistoricalIsLoading,
    error: allHistoricalError,
  } = useQuery({
    queryKey: ['all_historical_data', lstmGranularity], 
    queryFn: () => fetchAllHistoricalData(lstmGranularity),
    enabled: activeTab === 'lstm',
    retry: false,
  });

  // LSTM forecast query for all cities
  const {
    data: lstmData,
    isLoading: lstmIsLoading,
    error: lstmError,
    refetch: lstmRefetch,
    isFetching: lstmIsFetching
  } = useQuery({
    queryKey: ['lstm_forecasts', lstmGranularity], 
    queryFn: () => fetchLSTMForecasts(lstmGranularity),
    enabled: false,
    retry: false,
    onSuccess: (data) => console.log("✅ LSTM Forecasts received (frontend):", data),
    onError: (err) => console.error("Error fetching LSTM forecast data:", err.message)
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!selectedCity) {
      alert("Please select a city to get a forecast.");
      return;
    }
    refetch();
  };

  const handleLSTMSubmit = (e) => {
    e.preventDefault();
    lstmRefetch();
  };

  // Process single city forecast data with historical data
  let chartData = { labels: [], datasets: [] };
  if (selectedCity) {
    const historical = historicalData?.historical || [];
    const forecast = predictionData?.forecast || [];
    
    // Combine historical and forecast data
    const allLabels = [
      ...historical.map(item => item.date),
      ...forecast.map(item => item.date)
    ];
    
    const historicalValues = historical.map(item => item.transaction_value);
    const forecastValues = forecast.map(item => item.predicted_value);
    
    // Create datasets
    const datasets = [];
    
    // Historical data dataset
    if (historical.length > 0) {
      datasets.push({
        label: `Historical Data - ${selectedCity}`,
        data: [
          ...historicalValues,
          ...new Array(forecast.length).fill(null) // Fill with null for forecast period
        ],
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        tension: 0.1,
        pointStyle: 'circle',
      });
    }
    
    // Forecast data dataset
    if (forecast.length > 0) {
      datasets.push({
        label: `Forecast - ${selectedCity}`,
        data: [
          ...new Array(historical.length).fill(null), // Fill with null for historical period
          ...forecastValues
        ],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.1,
        pointStyle: 'triangle',
        borderDash: [5, 5], // Dashed line for predictions
      });
    }
    
    chartData = {
      labels: allLabels,
      datasets: datasets
    };
  }

  // Process LSTM forecast data for all cities with historical data
  let lstmChartData = { labels: [], datasets: [] };
  if (lstmData?.forecasts && allHistoricalData?.historical) {
    const colors = [
      'rgb(255, 99, 132)',
      'rgb(54, 162, 235)',
      'rgb(255, 205, 86)',
      'rgb(75, 192, 192)',
      'rgb(153, 102, 255)'
    ];
    
    const cities = Object.keys(lstmData.forecasts);
    const historicalByCities = allHistoricalData.historical;
    
    if (cities.length > 0) {
      // Get all unique dates (historical + forecast)
      const allDates = new Set();
      
      // Add historical dates
      Object.values(historicalByCities).forEach(cityData => {
        cityData.forEach(item => allDates.add(item.date));
      });
      
      // Add forecast dates
      Object.values(lstmData.forecasts).forEach(cityForecasts => {
        cityForecasts.forEach(item => allDates.add(item.date));
      });
      
      lstmChartData.labels = Array.from(allDates).sort();
      
      // Create datasets for each city (historical + forecast)
      lstmChartData.datasets = [];
      
      cities.forEach((city, index) => {
        const cityHistorical = historicalByCities[city] || [];
        const cityForecast = lstmData.forecasts[city] || [];
        
        const color = colors[index % colors.length];
        
        // Historical dataset
        if (cityHistorical.length > 0) {
          const historicalData = lstmChartData.labels.map(date => {
            const found = cityHistorical.find(item => item.date === date);
            return found ? found.transaction_value : null;
          });
          
          lstmChartData.datasets.push({
            label: `${city} (Historical)`,
            data: historicalData,
            borderColor: color,
            backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.3)'),
            tension: 0.1,
            pointStyle: 'circle',
          });
        }
        
        // Forecast dataset
        if (cityForecast.length > 0) {
          const forecastData = lstmChartData.labels.map(date => {
            const found = cityForecast.find(item => item.date === date);
            return found ? found.predicted_value : null;
          });
          
          lstmChartData.datasets.push({
            label: `${city} (LSTM Forecast)`,
            data: forecastData,
            borderColor: color,
            backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.5)'),
            tension: 0.1,
            pointStyle: 'triangle',
            borderDash: [5, 5], // Dashed line for predictions
          });
        }
      });
    }
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: activeTab === 'single' 
          ? 'Historical Data (2012-2016) + Future Forecast' 
          : 'LSTM Forecasts: Historical + Future for All Cities',
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      },
      x: {
        title: {
          display: true,
          text: 'Date'
        }
      }
    },
    interaction: {
      intersect: false,
      mode: 'index',
    },
  };

  const tabStyle = (isActive) => ({
    padding: '10px 20px',
    cursor: 'pointer',
    backgroundColor: isActive ? '#2C3E50' : '#ecf0f1',
    color: isActive ? 'white' : '#2C3E50',
    border: '1px solid #2C3E50',
    borderBottom: isActive ? 'none' : '1px solid #2C3E50',
    marginRight: '5px',
    borderRadius: '5px 5px 0 0'
  });

  return (
    <div className="prediction-container" style={{ display: 'wrap', alignItems: 'center', padding: '20px', width: '100%', borderRadius: '5px' }}>
      <div>
        <h1 style={{ 
          fontFamily:'Cantarell, "Open Sans", "Helvetica Neue", sans-serif',
          paddingTop:'20px',  
          fontSize: '24px', 
          marginBottom: '20px' 
        }}>
          Transaction Value Time Series Forecasting (Historical + Future)
        </h1>
      </div>

      {/* Tab Navigation */}
      <div style={{ marginBottom: '20px' }}>
        <button 
          style={tabStyle(activeTab === 'single')}
          onClick={() => setActiveTab('single')}
        >
          Single City (Historical + Forecast)
        </button>
        <button 
          style={tabStyle(activeTab === 'lstm')}
          onClick={() => setActiveTab('lstm')}
        >
          LSTM Multi-City (Historical + Forecast)
        </button>
      </div>

      {/* Single City Forecasting Tab */}
      {activeTab === 'single' && (
        <div>
          <section style={{ height: '150px', overflow: 'hidden', flexDirection: 'row' }}>
            <form onSubmit={handleSubmit} style={{ 
              paddingTop:'35px',
              flex: '1', 
              display: 'flex', 
              borderRadius:'5px', 
              border: '1px solid var(--background-light-gray-f)',
              width:'100%', 
              height:'400px', 
              fontSize:'20px'
            }}>
              <div className="form-group">
                <label htmlFor="selectedCity" style={{ 
                  paddingLeft: '10px', 
                  fontSize: '20px', 
                  fontFamily:'Cantarell, "Open Sans", "Helvetica Neue", sans-serif' 
                }}>
                  City:
                </label>
                <select 
                  style={{ 
                    backgroundColor:'#2C3E50',
                    padding: '8px', 
                    fontWeight: '400', 
                    color:'white',
                    marginLeft: '20px',
                    height: '35px',
                    width: '150px' 
                  }}
                  id="selectedCity" 
                  value={selectedCity} 
                  onChange={(e) => setSelectedCity(e.target.value)}
                >
                  <option value="">-- Select a City --</option>
                  {predefinedCities.slice(1).map((city) => (
                    <option key={city} value={city}>{city}</option>
                  ))}
                </select>
              </div>

              <div className="form-group" style={{paddingLeft: '90px'}}>
                <label htmlFor="granularity">Time Unit:</label>
                <select 
                  style={{ 
                    backgroundColor:'#2C3E50',
                    padding: '8px', 
                    fontWeight: '400', 
                    color:'white',
                    marginLeft: '20px',
                    height: '35px',
                    width: '150px' 
                  }} 
                  id="granularity" 
                  value={granularity} 
                  onChange={(e) => setGranularity(e.target.value)}
                >
                  <option value="M">Monthly</option>
                  <option value="Y">Yearly</option>
                </select>
              </div>

              <div style={{ paddingLeft: '50px' }}>
                <button 
                  type="submit" 
                  disabled={isLoading || isFetching || !selectedCity} 
                  style={{ 
                    flexDirection:'row',
                    padding: '8px 15px', 
                    cursor: (isLoading || isFetching || !selectedCity) ? 'not-allowed' : 'pointer' 
                  }}
                >
                  {isFetching ? 'Predicting...' : 'Get Historical + Forecast'}
                </button>
              </div>
            </form>
          </section>

          {(isFetching || historicalIsLoading) && <p>Loading historical data and predictions...</p>}
          {(error || historicalError) && (
            <p className="error" style={{ color: 'red' }}>
              Error: {error?.message || historicalError?.message}
            </p>
          )}

          {selectedCity && (chartData.datasets.length > 0) && !isFetching && !historicalIsLoading && (
            <div className="results" style={{  
              borderRadius:'5px', 
              border: '1px solid var(--background-light-gray-f)', 
              backgroundColor: 'var(--background-light-white-p)',
              boxShadow: '0px 1px 5px 1px var(--background-light-gray-p)',  
              transition: 'transform 0.225s ease-in-out' 
            }}>
              <p>
                Displaying data for: City: <strong>{selectedCity}</strong>, 
                Granularity: <strong>{granularity === 'M' ? 'Monthly' : 'Yearly'}</strong>
                <br />
                <span style={{ fontSize: '14px', color: '#666' }}>
                  Blue line: Historical data (2012-2016) | Red dashed line: Future predictions
                </span>
              </p>
              <div className="chart-container" style={{ width: '100%', height: '350px', margin: 'auto' }}>
                <Line options={chartOptions} data={chartData} />
              </div>
            </div>
          )}

          {selectedCity && !chartData.datasets.length && !isFetching && !historicalIsLoading && !error && !historicalError && (
            <p style={{ marginTop: '20px' }}>
              Historical data loaded. Click "Get Historical + Forecast" to add future predictions.
            </p>
          )}

          {!selectedCity && (
            <p style={{ marginTop: '20px' }}>
              Please select a city to view historical data and forecasts.
            </p>
          )}
        </div>
      )}

      {/* LSTM Multi-City Forecasting Tab */}
      {activeTab === 'lstm' && (
        <div>
          <section style={{ height: '150px', overflow: 'hidden', flexDirection: 'row' }}>
            <form onSubmit={handleLSTMSubmit} style={{ 
              paddingTop:'35px',
              flex: '1', 
              display: 'flex', 
              borderRadius:'5px', 
              border: '1px solid var(--background-light-gray-f)',
              width:'100%', 
              height:'400px', 
              fontSize:'20px'
            }}>
              <div className="form-group">
                <label htmlFor="lstmGranularity" style={{ 
                  paddingLeft: '10px', 
                  fontSize: '20px', 
                  fontFamily:'Cantarell, "Open Sans", "Helvetica Neue", sans-serif' 
                }}>
                  Time Unit:
                </label>
                <select 
                  style={{ 
                    backgroundColor:'#2C3E50',
                    padding: '8px', 
                    fontWeight: '400', 
                    color:'white',
                    marginLeft: '20px',
                    height: '35px',
                    width: '150px' 
                  }} 
                  id="lstmGranularity" 
                  value={lstmGranularity} 
                  onChange={(e) => setLstmGranularity(e.target.value)}
                >
                  <option value="M">Monthly (60 months)</option>
                  <option value="Y">Yearly (5 years)</option>
                </select>
              </div>

              <div style={{ paddingLeft: '50px' }}>
                <button 
                  type="submit" 
                  disabled={lstmIsLoading || lstmIsFetching || allHistoricalIsLoading} 
                  style={{ 
                    flexDirection:'row',
                    padding: '8px 15px', 
                    cursor: (lstmIsLoading || lstmIsFetching || allHistoricalIsLoading) ? 'not-allowed' : 'pointer' 
                  }}
                >
                  {(lstmIsFetching || allHistoricalIsLoading) ? 'Loading...' : 'Get Historical + LSTM Forecasts'}
                </button>
              </div>
            </form>
          </section>

          {(lstmIsFetching || allHistoricalIsLoading) && <p>Loading historical data and LSTM forecasts for all cities...</p>}
          {(lstmError || allHistoricalError) && (
            <p className="error" style={{ color: 'red' }}>
              Error: {lstmError?.message || allHistoricalError?.message}
            </p>
          )}

          {lstmData?.forecasts && !lstmIsFetching && !allHistoricalIsLoading && (
            <div className="results" style={{  
              borderRadius:'5px', 
              border: '1px solid var(--background-light-gray-f)', 
              backgroundColor: 'var(--background-light-white-p)',
              boxShadow: '0px 1px 5px 1px var(--background-light-gray-p)',  
              transition: 'transform 0.225s ease-in-out' 
            }}>
              <p>
                Displaying historical + LSTM forecasts for: 
                <strong> {Object.keys(lstmData.forecasts).length} cities</strong>, 
                Granularity: <strong>{lstmGranularity === 'M' ? 'Monthly' : 'Yearly'}</strong>,
                Forecast Periods: <strong>{lstmData.forecast_periods}</strong>
                <br />
                <span style={{ fontSize: '14px', color: '#666' }}>
                  Solid lines: Historical data (2012-2016) | Dashed lines: LSTM predictions
                </span>
              </p>
              <div className="chart-container" style={{ width: '100%', height: '350px', margin: 'auto' }}>
                <Line options={chartOptions} data={lstmChartData} />
              </div>
              
              {/* Summary Table */}
              <div style={{ marginTop: '20px', padding: '10px' }}>
                <h3>Forecast Summary</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '10px' }}>
                  <thead>
                    <tr style={{ backgroundColor: '#2C3E50', color: 'white' }}>
                      <th style={{ padding: '8px', border: '1px solid #ddd' }}>City</th>
                      <th style={{ padding: '8px', border: '1px solid #ddd' }}>First Prediction</th>
                      <th style={{ padding: '8px', border: '1px solid #ddd' }}>Last Prediction</th>
                      <th style={{ padding: '8px', border: '1px solid #ddd' }}>Average</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(lstmData.forecasts).map(([city, forecasts]) => {
                      const firstPred = forecasts[0]?.predicted_value || 0;
                      const lastPred = forecasts[forecasts.length - 1]?.predicted_value || 0;
                      const average = forecasts.reduce((sum, f) => sum + f.predicted_value, 0) / forecasts.length;
                      
                      return (
                        <tr key={city}>
                          <td style={{ padding: '8px', border: '1px solid #ddd' }}>{city}</td>
                          <td style={{ padding: '8px', border: '1px solid #ddd' }}>{firstPred.toFixed(2)}</td>
                          <td style={{ padding: '8px', border: '1px solid #ddd' }}>{lastPred.toFixed(2)}</td>
                          <td style={{ padding: '8px', border: '1px solid #ddd' }}>{average.toFixed(2)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {!lstmData?.forecasts && !lstmIsFetching && !lstmError && !allHistoricalIsLoading && !allHistoricalError && (
            <p style={{ marginTop: '20px' }}>
              Historical data loaded. Click "Get Historical + LSTM Forecasts" to add comprehensive forecasting predictions.
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default TimeSeriesForecasting;

