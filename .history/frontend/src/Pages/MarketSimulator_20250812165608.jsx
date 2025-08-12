import React, { useState, useEffect } from "react";
import { Line } from 'react-chartjs-2';
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
    const predefinedCities = ["Beirut", "Baabda, Aley, Chouf", "Kesrouan, Jbeil", "Tripoli, Akkar", "Bekaa"];
    const [selectedCity, setSelectedCity] = useState("Beirut");
    const [scenarioVisibility, setScenarioVisibility] = useState({
        baseline: true,
        boom: false,
        crash: false,
    });
    const [chartData, setChartData] = useState({ labels: [], datasets: [] });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchData = async (city) => {
        if (!city) return;
        setLoading(true);
        setError(null);
        try {
            const url = `http://127.0.0.1:8000/api/market-simulator?selection=${encodeURIComponent(city)}`;
            console.log("-> Fetching data from URL:", url);
            const response = await fetch(url);
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || `HTTP Error ${response.status}`);
            }
            const apiData = await response.json();
            
          
            const processedData = {
                labels: new Set(), 
                baseline: {},
                boom: {},
                crash: {}
            };

            apiData.forEach(item => {
                processedData.labels.add(item.date);
                
                processedData[item.scenario.toLowerCase()][item.date] = item.transaction_value;
            });
            
            
            const sortedLabels = Array.from(processedData.labels).sort();

            const datasets = [
                {
                    label: 'Baseline',
                    data: sortedLabels.map(label => processedData.baseline[label] || null),
                    borderColor: 'rgb(53, 162, 235)',
                    backgroundColor: 'rgba(53, 162, 235, 0.5)',
                    hidden: !scenarioVisibility.baseline,
                },
                {
                    label: 'Boom',
                    data: sortedLabels.map(label => processedData.boom[label] || null),
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.5)',
                    hidden: !scenarioVisibility.boom,
                },
                {
                    label: 'Crash',
                    data: sortedLabels.map(label => processedData.crash[label] || null),
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    hidden: !scenarioVisibility.crash,
                }
            ];

            setChartData({ labels: sortedLabels, datasets });

        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

  
    useEffect(() => {
        fetchData(selectedCity);
    }, []);

    const handleGetSimulation = () => {
       
        fetchData(selectedCity);
    };

    
   
    const handleCheckboxChange = (event) => {
        const { name, checked } = event.target;
        setScenarioVisibility(prevState => {
            const newState = { ...prevState, [name]: checked };

          
            setChartData(currentChartData => {
                const newDatasets = currentChartData.datasets.map(ds => {
                    if (ds.label.toLowerCase() === name) {
                        return { ...ds, hidden: !checked };
                    }
                    return ds;
                });
                return { ...currentChartData, datasets: newDatasets };
            });

            return newState;
        });
    };

    return (
    <div className="market-simulator-layout">
        <div className="market-simulator-content">
            <div className="market-simulator-title">
                <h1>Market Scenario Simulator</h1>
            </div>

           
            <div className="market-simulator-components">
                
                {/* Card for all the top controls */}
                <div className="controls-card">
                    <div className="controls-container">
                        
                        {/* Left side containing dropdown and checkboxes */}
                        <div className="input-groups">
                            <div className="form-group">
                                <label htmlFor="selectedCity">City</label>
                                <select id="selectedCity" value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)}>
                                    {predefinedCities.map(city => <option key={city} value={city}>{city}</option>)}
                                </select>
                            </div>

                            <div className="form-group">
                                <label>Scenarios</label>
                                <div className="scenario-group">
                                    <label className="checkbox-label">
                                        <input type="checkbox" name="baseline" checked={scenarioVisibility.baseline} onChange={handleCheckboxChange} />
                                        Baseline
                                    </label>
                                    <label className="checkbox-label">
                                        <input type="checkbox" name="boom" checked={scenarioVisibility.boom} onChange={handleCheckboxChange} />
                                        Boom
                                    </label>
                                    <label className="checkbox-label">
                                        <input type="checkbox" name="crash" checked={scenarioVisibility.crash} onChange={handleCheckboxChange} />
                                        Crash
                                    </label>
                                </div>
                            </div>
                        </div>

                        {/* Right side for the button */}
                        <div className="button-group">
                             <button type="button" onClick={handleGetSimulation} disabled={loading}>
                                {loading ? 'Loading...' : 'Get Simulation'}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Card specifically for the chart */}
                <div className="chart-card">
                    {loading && <p className="status-message">Loading chart...</p>}
                    {error && <p className="error-message">Error: {error}</p>}
                    {!loading && !error && chartData.labels.length > 0 && (
                        <div className="chart-wrapper">
                            <Line 
                                options={{ 
                                    responsive: true, 
                                    maintainAspectRatio: false 
                                }} 
                                data={chartData} 
                            />
                        </div>
                    )}
                </div>
                
            </div>
        </div>
    </div>
);
};

export default MarketSimulator;