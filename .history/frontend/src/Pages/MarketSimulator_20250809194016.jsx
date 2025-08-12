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
    const [selectedScenario, setSelectedScenario] = useState("");
    const [scenarioVisibility, setScenarioVisibility] = useState({
        baseline: true,
        boom: false,
        crash: false,
    });
    const [apiData, setApiData] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchData = async (city) => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`http://127.0.0.1:8000/v1/api/market-simulator?selection=${city}`);
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

    useEffect(() => {
        if (selectedCity) {
            fetchData(selectedCity);
        }
    }, [selectedCity]);

    const formatDataForChart = () => {
        if (!apiData || apiData.length === 0) return { labels: [], datasets: [] };

        const labels = [...new Set(apiData.map(item => item.date))].sort();
        
        const datasets = [];
        if (scenarioVisibility.baseline) {
            const baselineData = apiData.filter(d => d.scenario === 'Baseline');
            datasets.push({
                label: 'Baseline',
                data: baselineData.sort((a, b) => new Date(a.date) - new Date(b.date)).map(d => d.transaction_value),
                borderColor: 'rgb(53, 162, 235)',
                backgroundColor: 'rgba(53, 162, 235, 0.5)',
            });
        }
        if (scenarioVisibility.boom) {
            const boomData = apiData.filter(d => d.scenario === 'Boom');
            datasets.push({
                label: 'Boom Scenario',
                data: boomData.sort((a, b) => new Date(a.date) - new Date(b.date)).map(d => d.transaction_value),
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
            });
        }
        if (scenarioVisibility.crash) {
            const crashData = apiData.filter(d => d.scenario === 'Crash');
            datasets.push({
                label: 'Crash Scenario',
                data: crashData.sort((a, b) => new Date(a.date) - new Date(b.date)).map(d => d.transaction_value),
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
            title: { 
                display: true, 
                text: selectedCity ? `Market Scenarios for ${selectedCity}` : 'Select a city to view scenarios'
            },
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
                                id="selectedCity" 
                                value={selectedCity} 
                                onChange={(e) => setSelectedCity(e.target.value)}
                                required
                            >
                                <option value="">Select a City</option>
                                {predefinedCities.map((city) => (
                                    <option key={city} value={city}>{city}</option>
                                ))}
                            </select>
                        </div>

                        <div className="form-group">
                            <label htmlFor="scenario">Scenario</label>
                            <select 
                                id="scenario" 
                                value={selectedScenario} 
                                onChange={(e) => setSelectedScenario(e.target.value)}
                            >
                                <option value="">All Scenarios</option>
                                <option value="baseline">Baseline</option>
                                <option value="boom">Boom</option>
                                <option value="crash">Crash</option>
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
                            <button type="submit" disabled={loading || !selectedCity}>
                                {loading ? 'Loading...' : 'Get S'}
                            </button>
                        </div>
                    </form>

                    {loading && <p className="status-message">Loading forecast...</p>}
                    {error && <p className="error-message">Error: {error}</p>}

                    <div className="chart-container">
                        {loading ? (
                            <div className="loading-message">Loading chart...</div>
                        ) : error ? (
                            <div className="error-message">Error loading data</div>
                        ) : (
                            <Line options={chartOptions} data={chartData} />
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