import React from "react";
import { data } from "react-router-dom";
import "./BarChartComponent.css";
import {
  Bar,
  CartesianGrid,
  Legend,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const BarChartComponent = () => {
  const data = [
    { name: "a", value: 12 },
    { name: "b", value: 10 },
    { name: "c", value: 1 },
    { name: "d", value: 3 },
    { name: "e", value: 7 },
    { name: "a", value: 12 },
    { name: "b", value: 10 },
    { name: "c", value: 1 },
    { name: "d", value: 3 },
    { name: "e", value: 7 },
    { name: "a", value: 12 },
    { name: "b", value: 10 },
    { name: "c", value: 1 },
    { name: "d", value: 3 },
    { name: "e", value: 7 },
    { name: "a", value: 12 },
    { name: "b", value: 10 },
    { name: "c", value: 1 },
    { name: "d", value: 3 },
    { name: "e", value: 7 },
    { name: "a", value: 12 },
    { name: "b", value: 10 },
    { name: "c", value: 1 },
    { name: "d", value: 3 },
  ];

  return (
    <div className="barchart-content">
      <ResponsiveContainer width={"100%"} height={"100%"}>
        <BarChart
          data={data}
          margin={{ top: 20, right: 20, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="value" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default BarChartComponent;
