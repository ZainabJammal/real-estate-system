import React, { useEffect, useState } from "react";
import "./LineChartComponent.css";
import {
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Line,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

function LineChartComponent({ data = null }) {
  const [selectedCity, setSelectedCity] = useState("");
  const [chartData, setChartData] = useState([]);
  const [cities, setCities] = useState([]);

  useEffect(() => {
    if (!data || data.length === 0) {
      setChartData(
        ...[
          { date: "1", transaction_value: "0" },
          { date: "2", transaction_value: "10" },
          { date: "3", transaction_value: "20" },
          { date: "4", transaction_value: "30" },
          { date: "5", transaction_value: "40" },
        ]
      );
      return;
    }

    // Extract unique cities
    const uniqueCities = [...new Set(data.map((t) => t.city))];
    setCities(uniqueCities);

    // Set default city only when it hasn’t been selected yet
    if (!selectedCity && uniqueCities.length > 0) {
      setSelectedCity(uniqueCities[0]);
    }
  }, [data]);

  useEffect(() => {
    if (!data || !selectedCity) return;

    // Group transactions by month/year and sum transaction_value
    const grouped = data?.reduce((acc, transaction) => {
      const monthYear = transaction.date; // "mm/yyyy"
      const city = transaction.city;
      const value = parseFloat(transaction.transaction_value) || 0; // Ensure it's a number

      if (!acc[city]) acc[city] = {};
      if (!acc[city][monthYear]) acc[city][monthYear] = 0;

      acc[city][monthYear] += value; // Sum transaction values
      return acc;
    }, {});

    // Format data for Recharts (sorting by date)
    const formattedData = Object.entries(grouped[selectedCity] || {})
      .map(([month, totalValue]) => ({
        month,
        totalValue,
      }))
      .sort((a, b) => {
        // Sort months properly by year & month
        const [mA, yA] = a.month.split("/").map(Number);
        const [mB, yB] = b.month.split("/").map(Number);
        return yA === yB ? mA - mB : yA - yB;
      });

    setChartData(formattedData);
  }, [data, selectedCity]);

  // let data_ = [];
  // if (data != null) {
  //   // Aggregate transaction values by year
  //   const yearlyData = data?.reduce((acc, { date, transaction_value }) => {
  //     const year = date.split("/")[1]; // Extract year from "MM/YYYY"
  //     if (!acc[year]) {
  //       acc[year] = { total: 0, count: 0 };
  //     }
  //     acc[year].total += transaction_value;
  //     acc[year].count += 1;
  //     return acc;
  //   }, {});

  //   // Convert aggregated data into an array
  //   data_ = Object.entries(yearlyData).map(([year, { total, count }]) => ({
  //     date: year, // Use year as the date
  //     transaction_value: total / (count * count), // Compute the average
  //   }));
  // } else {
  //   data_ = [
  //     { date: "1", transaction_value: "0" },
  //     { date: "2", transaction_value: "10" },
  //     { date: "3", transaction_value: "20" },
  //     { date: "4", transaction_value: "30" },
  //     { date: "5", transaction_value: "40" },
  //   ];
  // }
  return (
    <>
      <label>Select City: </label>
      <select
        onChange={(e) => setSelectedCity(e.target.value)}
        value={selectedCity}
      >
        {[...new Set(data?.map((t) => t.city))].map((city) => (
          <option key={city} value={city}>
            {city}
          </option>
        ))}
      </select>

      <ResponsiveContainer width={"100%"} height={400}>
        <LineChart
          data={chartData}
          margin={{ top: 50, right: 50, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="1 1" />
          <XAxis dataKey="month" />
          <YAxis />
          <Tooltip />
          <Legend />

          <Line type="monotone" dataKey="totalValue" stroke="#fe4533" />
          {/* <Line type="monotone" dataKey="Agent2" stroke="#da829d" /> */}
        </LineChart>
      </ResponsiveContainer>
    </>
  );
}

export default LineChartComponent;
