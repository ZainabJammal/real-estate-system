import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const MarketScenarioChart = () => {
    // State to hold the data fetched from the Quart API
    const [scenarioData, setScenarioData] = useState([]);
    // State to manage loading status
    const [loading, setLoading] = useState(true);

    // Fetch data from the backend when the component mounts
    useEffect(() => {
        const fetchData = async () => {
            try {
                // Your Quart app is likely running on port 5000
                const response = await fetch('http://127.0.0.1:5000/api/market-scenarios');
                const data = await response.json();
                
                // Pivot the data for the charting library
                // We want one entry per date, with each scenario as a key
                const formattedData = {};
                data.forEach(row => {
                    if (!formattedData[row.date]) {
                        formattedData[row.date] = { date: row.date };
                    }
                    formattedData[row.date][row.scenario] = row.price;
                });

                setScenarioData(Object.values(formattedData));
                setLoading(false);
            } catch (error) {
                console.error("Failed to fetch scenario data:", error);
                setLoading(false);
            }
        };

        fetchData();
    }, []); // The empty array [] ensures this effect runs only once

    if (loading) {
        return <div>Loading Market Scenarios...</div>;
    }

    return (
        <div style={{ width: '100%', height: 400 }}>
            <h3>Market Scenarios (2017-2021 Simulated)</h3>
            <ResponsiveContainer>
                <LineChart
                    data={scenarioData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis 
                        tickFormatter={(value) => new Intl.NumberFormat('en-US', { notation: 'compact', compactDisplay: 'short' }).format(value)} 
                    />
                    <Tooltip 
                        formatter={(value) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)} 
                    />
                    <Legend />
                    <Line type="monotone" dataKey="Baseline" stroke="#8884d8" dot={false} />
                    <Line type="monotone" dataKey="Boom" stroke="#82ca9d" dot={false} />
                    <Line type="monotone" dataKey="Crash" stroke="#ff4d4d" dot={false} />
                </LineChart>
            </ResponsiveContainer>
            <p style={{ textAlign: 'center', fontSize: '0.8rem', color: '#666' }}>
                Note: 2017–2021 data represents simulated market scenarios for demonstration purposes and does not reflect actual historical prices.
            </p>
        </div>
    );
};

export default MarketScenarioChart;