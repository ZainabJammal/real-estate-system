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
    // --- STATE ---
    const predefinedCities = ["Baabda, Aley, Chouf", "Beirut", "Bekaa", "Kesrouan, Jbeil", "Tripoli, Akkar"];
    const [selectedCity, setSelectedCity] = useState("Beirut"); // Set a default to avoid empty state
    const [scenarioVisibility, setScenarioVisibility] = useState({
        baseline: true,
        boom: false,
        crash: false,
    });
    const [chartData, setChartData] = useState({ labels: [], datasets: [] }); // Default to empty chart structure
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // --- DATA FETCHING (Now only called by submit handler) ---
    const fetchData = async (city) => {
        setLoading(true);
        setError(null);
        setChartData({ labels: [], datasets: [] }); // Clear previous chart data
        try {
            const response = await fetch(`http://127.0.0.1:8000/v1/api/market-simulator?selection=${city}`);

              console.log("-> Fetching data from URL:", url);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || `HTTP error! status: ${response.status}`);
            }
            const apiData = await response.json();
            
            // --- Format data directly here after fetch ---
            const labels = [...new Set(apiData.map(item => item.date))].sort();
            const datasets = [];
            // (Your existing formatting logic is good, just move it here)
            if (scenarioVisibility.baseline) {
                datasets.push({
                    label: 'Baseline',
                    data: apiData.filter(d => d.scenario === 'Baseline').map(d => d.transaction_value),
                    borderColor: 'rgb(53, 162, 235)',
                });
            }
            if (scenarioVisibility.boom) {
                 datasets.push({
                    label: 'Boom Scenario',
                    data: apiData.filter(d => d.scenario === 'Boom').map(d => d.transaction_value),
                    borderColor: 'rgb(75, 192, 192)',
                });
            }
            if (scenarioVisibility.crash) {
                 datasets.push({
                    label: 'Crash Scenario',
                    data: apiData.filter(d => d.scenario === 'Crash').map(d => d.transaction_value),
                    borderColor: 'rgb(255, 99, 132)',
                });
            }
            setChartData({ labels, datasets });

        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // REMOVED the useEffect hook that was causing premature fetches.

    const handleSubmit = (e) => {
        e.preventDefault(); // Prevent page reload
        if (selectedCity) {
            fetchData(selectedCity);
        } else {
            setError("Please select a city to get a forecast.");
        }
    };


    const handleCheckboxChange = (event) => {
        const { name, checked } = event.target;
        setScenarioVisibility(prevState => ({ ...prevState, [name]: checked }));
    };

    return (
        <div className="market-simulator-layout">
            <div className="market-simulator-content">
                <div className="market-simulator-title">
                    <h1>Market Scenario Simulator</h1>
                </div>
                <div className="dashboard-components">
                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label htmlFor="selectedCity">City</label>
                            <select 
                                value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)}>
                                {predefinedCities.map(city => <option key={city} value={city}>{city}</option>)}
                              
                            </select>
                        </div>

                        <div className="form-group checkbox-group">
                            <label>
                                <input 
                                    type="checkbox" 
                                    name="baseline" 
                                    checked={scenarioVisibility.baseline} 
                                    onChange={handleCheckboxChange} 
                                /> Baseline
                            </label>
                            <label>
                                <input 
                                    type="checkbox" 
                                    name="boom" 
                                    checked={scenarioVisibility.boom} 
                                    onChange={handleCheckboxChange} 
                                /> Boom
                            </label>
                            <label>
                                <input 
                                    type="checkbox" 
                                    name="crash" 
                                    checked={scenarioVisibility.crash} 
                                    onChange={handleCheckboxChange} 
                                /> Crash
                            </label>
                        </div>

                        <div className="form-group">
                            <button onClick={() => fetchData(selectedCity)} disabled={loading}>
                                {loading ? 'Loading...' : 'Get Scenario Simulation'}
                            </button>
                        </div>
                    </form>

                    {loading && <p className="status-message">Loading forecast...</p>}
                    {error && <p className="error-message">Error: {error}</p>}

                   <div style={{ position: 'relative', height: '400px' }}>
                {loading && <p>Loading...</p>}
                {error && <p style={{color: 'red'}}>Error: {error}</p>}
                {!loading && !error && chartData.datasets.length > 0 && (
                    <Line 
                        options={{ responsive: true, maintainAspectRatio: false }} 
                        data={chartData} 
                    />
                )}
            </div>
                    <p className="disclaimer">
                        Note: 2017–2021 data represents simulated scenarios for demonstration purposes.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default MarketSimulator;