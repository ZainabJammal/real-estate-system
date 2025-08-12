
// npm install recharts @chakra-ui/react @emotion/react @emotion/styled framer-motion
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardBody, Heading, Select, Checkbox, HStack, Spinner, Center, Text } from '@chakra-ui/react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// --- The Main Component ---
const MarketSimulator = () => {
    const predefinedCities = ["Baabda, Aley, Chouf", "Beirut", "Bekaa", "Kesrouan, Jbeil", "Tripoli, Akkar"];
    const [selectedCity, setSelectedCity] = useState("");
    const [scenarioVisibility, setScenarioVisibility] = useState({
        baseline: true,
        boom: false,
        crash: false,
    });

    // State for data and API status
    const [chartData, setChartData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // --- DATA FETCHING ---
    useEffect(() => {
        // This function fetches data when the component mounts or when selectedCity changes
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                // Fetch all 3 scenarios for the selected city at once
                const response = await fetch(`http://127.0.0.1:8000/v1/api/market-simulator?selection=${selectedCity}`);
                if (!response.ok) {
                    throw new Error(`Failed to fetch data: ${response.statusText}`);
                }
                const data = await response.json();
                
                // --- Data Transformation for Recharts ---
                // We pivot the data to make it easy for the chart to read
                const formattedData = {};
                data.forEach(row => {
                    if (!formattedData[row.date]) {
                        formattedData[row.date] = { date: row.date };
                    }
                    // e.g., formattedData['2017-01-01']['Baseline'] = 241.931
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
    }, [selectedCity]); // The dependency array ensures this runs whenever selectedCity changes


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
                    <Select value={predefinedCities} onChange={(e) => setSelectedCity(e.target.value)} maxWidth="200px">
                        <option value="predefinedCities">Select a city</option>Beirut</option>
                        <option value="Baabda">Baabda, Aley, Chouf</option>
                        <option value="Kesrouan">Kesrouan, Jbeil</option>
                        <option value="Tripoli">Tripoli, Akkar</option>
                        <option value="Bekaa">Bekaa</option>
                    </Select>

                    <Checkbox name="baseline" isChecked={scenarioVisibility.baseline} onChange={handleCheckboxChange}>
                        Baseline
                    </Checkbox>
                    <Checkbox name="boom" isChecked={scenarioVisibility.boom} onChange={handleCheckboxChange} colorScheme="green">
                        Boom Scenario
                    </Checkbox>
                    <Checkbox name="crash" isChecked={scenarioVisibility.crash} onChange={handleCheckboxChange} colorScheme="red">
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
                                {/* A line is only rendered if its corresponding checkbox is true */}
                                {scenarioVisibility.baseline && <Line type="monotone" dataKey="Baseline" stroke="#8884d8" dot={false} />}
                                {scenarioVisibility.boom && <Line type="monotone" dataKey="Boom" stroke="#2f855a" dot={false} />}
                                {scenarioVisibility.crash && <Line type="monotone" dataKey="Crash" stroke="#c53030" dot={false} />}
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