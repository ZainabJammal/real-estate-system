// This is the full, final code for your React component.
// It will call the now-working URL: /api/market-simulator
import React, { useState, useEffect } from "react";
import { Line } from 'react-chartjs-2';
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
    const predefinedCities = ["Beirut", "Baabda, Aley, Chouf", "Kesrouan, Jbeil", "Tripoli, Akkar", "Bekaa"];
    const [selectedCity, setSelectedCity] = useState("Beirut");
    const [scenarioVisibility, setScenarioVisibility] = useState({
        baseline: true,
        boom: false,
        crash: false,
    });
    const [chartData, setChartData] = useState({ labels: [], datasets: [] });
    const [loading, setLoading] = useState(true); // Start loading on initial mount
    const [error, setError] = useState(null);

    // --- Data Fetching Function ---
    const fetchData = async (city) => {
        if (!city) return;
        setLoading(true);
        setError(null);
        try {
            const url = `http://127.0.0.1:8000/api/market-simulator?selection=${city}`;
            console.log("-> Fetching data from URL:", url);
            const response = await fetch(url);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || `HTTP Error ${response.status}`);
            }
            const apiData = await response.json();
            
            // Format data for Chart.js
            const labels = [...new Set(apiData.map(item => item.date))].sort();
            const datasets = [];
            
            datasets.push({
                label: 'Baseline',
                data: labels.map(label => apiData.find(d => d.date === label && d.scenario === 'Baseline')?.transaction_value || null),
                borderColor: 'rgb(53, 162, 235)',
                hidden: !scenarioVisibility.baseline, // Control visibility via dataset property
            });
            datasets.push({
                label: 'Boom',
                data: labels.map(label => apiData.find(d => d.date === label && d.scenario === 'Boom')?.transaction_value || null),
                borderColor: 'rgb(75, 192, 192)',
                hidden: !scenarioVisibility.boom,
            });
            datasets.push({
                label: 'Crash',
                data: labels.map(label => apiData.find(d => d.date === label && d.scenario === 'Crash')?.transaction_value || null),
                borderColor: 'rgb(255, 99, 132)',
                hidden: !scenarioVisibility.crash,
            });

            setChartData({ labels, datasets });

        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // --- Fetch data on initial component mount ---
    useEffect(() => {
        fetchData(selectedCity);
    }, []);

    // --- Event Handlers ---
    const handleGetForecast = () => {
        fetchData(selectedCity);
    };

    const handleCheckboxChange = (event) => {
        const { name, checked } = event.target;
        setScenarioVisibility(prevState => ({ ...prevState, [name]: checked }));

        // Update chart visibility without refetching data
        const newChartData = { ...chartData };
        const datasetToUpdate = newChartData.datasets.find(ds => ds.label.toLowerCase().startsWith(name));
        if (datasetToUpdate) {
            datasetToUpdate.hidden = !checked;
            setChartData(newChartData);
        }
    };

    // --- Render Logic ---
    return (
        <div >
            <h1>Market Scenario Simulator</h1>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', margin: '1rem 0' }}>
                <select value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)}>
                    {predefinedCities.map(city => <option key={city} value={city}>{city}</option>)}
                </select>
                <button onClick={handleGetForecast} disabled={loading}>
                    {loading ? 'Loading...' : 'Get Forecast'}
                </button>
            </div>
            <div style={{ display: 'flex', gap: '1rem', margin: '1rem 0' }}>
                <label><input type="checkbox" name="baseline" checked={scenarioVisibility.baseline} onChange={handleCheckboxChange} /> Baseline</label>
                <label><input type="checkbox" name="boom" checked={scenarioVisibility.boom} onChange={handleCheckboxChange} /> Boom</label>
                <label><input type="checkbox" name="crash" checked={scenarioVisibility.crash} onChange={handleCheckboxChange} /> Crash</label>
            </div>
            <div style={{ position: 'relative', height: '400px' }}>
                {loading && <p>Loading chart...</p>}
                {error && <p style={{ color: 'red' }}>Error: {error}</p>}
                {!loading && !error && <Line options={{ responsive: true, maintainAspectRatio: false }} data={chartData} />}
            </div>
        </div>
    );
};

export default MarketSimulator;