
// npm install recharts @chakra-ui/react @emotion/react @emotion/styled framer-motion
// import React, { useState, useEffect } from 'react';
// import { Card, CardHeader, CardBody, Heading, Select, Checkbox, HStack, Spinner, Center, Text } from '@chakra-ui/react';

// // --- CORRECTED IMPORTS ---
// // Import ResponsiveContainer separately and other components together.
// import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { Line } from 'react-chartjs-2';
import React, { useState } from "react";

// --- The Main Component ---
const MarketSimulator = () => {
    // --- STATE MANAGEMENT ---
    // State for user selections
    const [selectedCity, setSelectedCity] = useState('Beirut');
    const [scenarioVisibility, setScenarioVisibility] = useState({
        Baseline: true, // Use PascalCase to match dataKey
        Boom: false,    // Use PascalCase to match dataKey
        Crash: false,   // Use PascalCase to match dataKey
    });

    // State for data and API status
    const [chartData, setChartData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // --- DATA FETCHING ---
    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                const response = await fetch(`http://127.0.0.1:8000/v1/api/market-simulator?selection=${selectedCity}`);
                if (!response.ok) {
                    throw new Error(`Failed to fetch data: ${response.statusText}`);
                }
                const data = await response.json();
                
                const formattedData = {};
                data.forEach(row => {
                    if (!formattedData[row.date]) {
                        formattedData[row.date] = { date: row.date };
                    }
                    formattedData[row.date][row.scenario] = row.transaction_value;
                });
                
                setChartData(Object.values(formattedData));

            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [selectedCity]);


    // --- EVENT HANDLERS ---
    const handleCheckboxChange = (event) => {
        const { name, checked } = event.target;
        setScenarioVisibility(prevState => ({
            ...prevState,
            [name]: checked,
        }));
    };

    // --- RENDER LOGIC ---
    return (
        <Card borderWidth="1px" borderRadius="lg" p={4} boxShadow="md">
            <CardHeader>
                <Heading size="md">Market Scenario Simulator</Heading>
            </CardHeader>

            <CardBody>
                {/* --- UI Controls --- */}
                <HStack spacing={8} mb={6}>
                    <Select value={selectedCity} onChange={(e) => setSelectedCity(e.target.value)} maxWidth="200px">
                        <option value="Beirut">Beirut</option>
                        <option value="Baabda">Baabda, Aley, Chouf</option>
                        <option value="Kesrouan">Kesrouan, Jbeil</option>
                        <option value="Tripoli">Tripoli, Akkar</option>
                        <option value="Bekaa">Bekaa</option>
                    </Select>
                    
                    {/* Make sure the 'name' attribute matches the keys in scenarioVisibility state */}
                    <Checkbox name="Baseline" isChecked={scenarioVisibility.Baseline} onChange={handleCheckboxChange}>
                        Baseline
                    </Checkbox>
                    <Checkbox name="Boom" isChecked={scenarioVisibility.Boom} onChange={handleCheckboxChange} colorScheme="green">
                        Boom Scenario
                    </Checkbox>
                    <Checkbox name="Crash" isChecked={scenarioVisibility.Crash} onChange={handleCheckboxChange} colorScheme="red">
                        Crash Scenario
                    </Checkbox>
                </HStack>

                {/* --- Chart Area (with loading/error states) --- */}
                <div style={{ width: '100%', height: '400px' }}>
                    {loading ? (
                        <Center h="100%"><Spinner size="xl" /></Center>
                    ) : error ? (
                        <Center h="100%"><Text color="red.500">Error: {error}</Text></Center>
                    ) : (
                        <ResponsiveContainer>
                            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                                <YAxis tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`} tick={{ fontSize: 12 }} />
                                <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
                                <Legend />

                                {/* --- CONDITIONAL LINE RENDERING --- */}
                                {/* The dataKey MUST EXACTLY match the keys in your formatted data */}
                                {scenarioVisibility.Baseline && <Line type="monotone" dataKey="Baseline" stroke="#8884d8" dot={false} />}
                                {scenarioVisibility.Boom && <Line type="monotone" dataKey="Boom" stroke="#2f855a" dot={false} />}
                                {scenarioVisibility.Crash && <Line type="monotone" dataKey="Crash" stroke="#c53030" dot={false} />}
                            </LineChart>
                        </ResponsiveContainer>
                    )}
                </div>
                <Text fontSize="xs" color="gray.500" mt={2} textAlign="center">
                    Note: 2017–2021 data represents simulated scenarios for demonstration purposes.
                </Text>
            </CardBody>
        </Card>
    );
};

export default MarketSimulator;