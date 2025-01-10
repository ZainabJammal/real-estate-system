import React, { useEffect, useState } from "react";
import "./Custom.css";
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

function CustomL({ title, desc }) {
  const data = [
    { months: "Jan", Agent1: 0, Agent2: 2 },
    { months: "Feb", Agent1: 7, Agent2: 5 },
    { months: "Mar", Agent1: 10, Agent2: 15 },
    { months: "Apr", Agent1: 20, Agent2: 18 },
    { months: "Jun", Agent1: 35, Agent2: 27 },
  ];

  return (
    <div className="card-layout">
      <div className="card-content">
        <div className="card-title">
          <h3>{title}</h3>
          <p>{desc}</p>
        </div>
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
    </div>
  );
}

export default CustomL;
