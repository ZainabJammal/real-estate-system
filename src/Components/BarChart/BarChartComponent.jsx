import React, { useEffect, useState } from "react";
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

const BarChartComponent = ({ data, size }) => {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (data) {
      setChartData(
        data?.map((prop) => ({
          name: prop?.province ? prop?.province : prop?.district,
          median_price: prop?.median_price_$,
          avg_price: prop?.avg_price_$,
          min_price: prop?.min_price_$,
        }))
      );
    }
  }, [data]);

  return (
    <>
      <ResponsiveContainer width={"100%"} height={380}>
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 20, left: 20, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="1 1" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />

          <Bar dataKey="median_price" fill="#52a832" />
          <Bar dataKey="avg_price" fill="#e22368" />
          <Bar dataKey="min_price" fill="#3275f8" />
        </BarChart>
      </ResponsiveContainer>
    </>
  );
};

export default BarChartComponent;
