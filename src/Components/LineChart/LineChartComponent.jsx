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

function LineChartComponent() {
  const data = [
    { months: "Jan", Agent1: 0, Agent2: 2 },
    { months: "Feb", Agent1: 7, Agent2: 5 },
    { months: "Mar", Agent1: 10, Agent2: 15 },
    { months: "Apr", Agent1: 20, Agent2: 18 },
    { months: "Jun", Agent1: 35, Agent2: 27 },
  ];
  return (
    <div>
      <ResponsiveContainer width={500} height={400}>
        <LineChart
          data={data}
          margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="1 1" />
          <XAxis dataKey="months" />
          <YAxis />
          <Tooltip />
          <Legend />

          <Line type="monotone" dataKey="Agent1" stroke="#82ca9d" />
          <Line type="monotone" dataKey="Agent2" stroke="#da829d" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default LineChartComponent;
