import React from "react";
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

function LineChartComponent({ data = [] }) {
  let data_ = [];
  if (data != null) {
    // Aggregate transaction values by year
    const yearlyData = data?.reduce((acc, { date, transaction_value }) => {
      const year = date.split("/")[1]; // Extract year from "MM/YYYY"
      if (!acc[year]) {
        acc[year] = { total: 0, count: 0 };
      }
      acc[year].total += transaction_value;
      acc[year].count += 1;
      return acc;
    }, {});

    // Convert aggregated data into an array
    data_ = Object.entries(yearlyData).map(([year, { total, count }]) => ({
      date: year, // Use year as the date
      transaction_value: total / (count * count), // Compute the average
    }));
  } else {
    data_ = [
      { date: "1", transaction_value: "0" },
      { date: "2", transaction_value: "10" },
      { date: "3", transaction_value: "20" },
      { date: "4", transaction_value: "30" },
      { date: "5", transaction_value: "40" },
    ];
  }
  return (
    <ResponsiveContainer width={"100%"} height={"100%"}>
      <LineChart
        data={data_}
        margin={{ top: 10, right: 50, left: 0, bottom: 10 }}
      >
        <CartesianGrid strokeDasharray="1 1" />
        <XAxis dataKey="date" />
        <YAxis dataKey="transaction_value" />
        <Tooltip />
        <Legend />

        <Line type="monotone" dataKey="transaction_value" stroke="#82ca9d" />
        {/* <Line type="monotone" dataKey="Agent2" stroke="#da829d" /> */}
      </LineChart>
    </ResponsiveContainer>
  );
}

export default LineChartComponent;
