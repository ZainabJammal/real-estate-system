import { useQuery } from '@tanstack/react-query';
import { Line } from 'react-chartjs-2';
import React, { useState, useEffect } from "react";
import "./MarketSimulator.css"; 

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


const MarketSimulator = () => {
    const predefinedCities = ["Baabda, Aley, Chouf", "Beirut", "Bekaa", "Kesrouan, Jbeil", "Tripoli, Akkar"];
    const [selectedCity, setSelectedCity] = useState("");
    const [scenarioVisibility, setScenarioVisibility] = useState({
        baseline: true,
        boom: false,
        crash: false,
    });
    const [apiData, setApiData] = useState([]); // Store raw API data
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // --- DATA FETCHING (No changes here) ---
    useEffect(() => {
        const fetchData = async (city) => {
            setLoading(true);
            setError(null);
            try {
                const response = await fetch(`http://127.0.0.1:8000/v1/api/market-simulator?selection=${selectedCity}`);
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
                    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                setApiData(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        fetchData();  // Call the fetchData function with the selected city
    }, [selectedCity]);

    const formatDataForChart = () => {
        const labels = [...new Set(apiData.map(item => item.date))].sort();
        
        const datasets = [];

        if (scenarioVisibility.baseline) {
            datasets.push({
                label: 'Baseline',
                data: apiData.filter(d => d.scenario === 'Baseline').map(d => d.transaction_value),
                borderColor: 'rgb(53, 162, 235)',
                backgroundColor: 'rgba(53, 162, 235, 0.5)',
            });
        }
        if (scenarioVisibility.boom) {
            datasets.push({
                label: 'Boom Scenario',
                data: apiData.filter(d => d.scenario === 'Boom').map(d => d.transaction_value),
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
            });
        }
        if (scenarioVisibility.crash) {
            datasets.push({
                label: 'Crash Scenario',
                data: apiData.filter(d => d.scenario === 'Crash').map(d => d.transaction_value),
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
            });
        }
        
        return { labels, datasets };
    };

    const chartData = formatDataForChart();

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { position: 'top' },
            title: { display: true, text: `Market Scenarios for ${selectedCity}` },
        },
    };
    const handleSubmit = (e) => {
    e.preventDefault();
    if (selectedCity) {
      fetchData(selectedCity);
    } else {
      alert("Please select a city to get a forecast.");
    }
  };

    // --- EVENT HANDLERS (No changes here) ---
    const handleCheckboxChange = (event) => {
        const { name, checked } = event.target;
        setScenarioVisibility(prevState => ({ ...prevState, [name]: checked }));
    };

    // --- RENDER LOGIC ---
    return (
      <div className="market-simulator-layout">
      <div className="market-simulator-content">
        <div className="forecasting-title" style={{ fontSize: '20px !important' }}>
          <h1>Transaction Value Forecasting</h1>
        </div>
        <div style={{ border: '1px solid #e2e8f0', borderRadius: '8px', padding: '16px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)' }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>Market Scenario Simulator</h3>
            
            <div style={{ display: 'flex', gap: '24px', margin: '16px 0' }}>
                <select value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)} style={{ padding: '8px' }}>
                    <option value="Beirut">Beirut</option>
                    <option value="Baabda">Baabda, Aley, Chouf</option>
                    <option value="Kesrouan">Kesrouan, Jbeil</option>
                    <option value="Tripoli">Tripoli, Akkar</option>
                    <option value="Bekaa">Bekaa</option>
                </select>

                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                    <label><input type="checkbox" name="baseline" checked={scenarioVisibility.baseline} onChange={handleCheckboxChange} /> Baseline</label>
                    <label><input type="checkbox" name="boom" checked={scenarioVisibility.boom} onChange={handleCheckboxChange} /> Boom</label>
                    <label><input type="checkbox" name="crash" checked={scenarioVisibility.crash} onChange={handleCheckboxChange} /> Crash</label>
                </div>
            </div>

            <div style={{ position: 'relative', height: '400px' }}>
                {loading && <div style={{ textAlign: 'center' }}>Loading...</div>}
                {error && <div style={{ color: 'red', textAlign: 'center' }}>Error: {error}</div>}
                {!loading && !error && <Line options={chartOptions} data={chartData} />}
            </div>
            <p style={{ fontSize: '0.75rem', color: '#718096', marginTop: '8px', textAlign: 'center' }}>
                Note: 2017–2021 data represents simulated scenarios for demonstration purposes.
            </p>
        </div>
    );
};

export default MarketSimulator;