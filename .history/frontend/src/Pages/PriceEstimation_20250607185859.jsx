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







  const fetchPriceEstimation = async (cityForApi, granularityForApi) => {
    const params = {
      city_name: cityForApi === "" ? null : cityForApi,
      granularity: granularityForApi,
    };
    console.log("Attempting to fetch Current Price Estimation with params:", params);
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
