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

const BarChartComponent = ({ data }) => {
  // const data = [
  //   { name: "a", value: 12 },
  //   { name: "b", value: 10 },
  //   { name: "c", value: 1 },
  //   { name: "d", value: 3 },
  //   { name: "e", value: 7 },
  //   { name: "a", value: 12 },
  //   { name: "b", value: 10 },
  //   { name: "c", value: 1 },
  //   { name: "d", value: 3 },
  //   { name: "e", value: 7 },
  //   { name: "a", value: 12 },
  //   { name: "b", value: 10 },
  //   { name: "c", value: 1 },
  //   { name: "d", value: 3 },
  //   { name: "e", value: 7 },
  //   { name: "a", value: 12 },
  //   { name: "b", value: 10 },
  //   { name: "c", value: 1 },
  //   { name: "d", value: 3 },
  //   { name: "e", value: 7 },
  //   { name: "a", value: 12 },
  //   { name: "b", value: 10 },
  //   { name: "c", value: 1 },
  //   { name: "d", value: 3 },
  // ];

  const chartData = data?.map((prop) => ({
    name: prop?.province ? prop?.province : prop?.district,
    max_price: prop?.max_price_$,
    min_price: prop?.min_price_$,
    avg_price: prop?.avg_price_$,
  }));

  return (
    <>
      <ResponsiveContainer width={"100%"} height={400}>
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 20, left: 20, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="1 1" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="max_price" fill="#a83232" />
          <Bar dataKey="avg_price" fill="#32a832" />
          <Bar dataKey="min_price" fill="#3232d8" />
        </BarChart>
      </ResponsiveContainer>
    </>
  );
};

export default BarChartComponent;
