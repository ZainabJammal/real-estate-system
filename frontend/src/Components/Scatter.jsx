import React, { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
} from "recharts";

const ScatterComp = ({ data }) => {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (data) {
      setChartData(
        data?.map((prop) => ({
          city: prop?.city,
          size_m2: prop?.size_m2,
          price_$: prop?.price_$,
          price_in_m2: prop?.size_m2
            ? parseFloat((prop.price_$ / prop.size_m2).toFixed(2))
            : 0,
          type: prop?.type,
        }))
      );
    }
  }, [data]);

  console.log(chartData);

  return (
    <>
      <ResponsiveContainer width="100%" height={380}>
        <ScatterChart
          style={{ fontSize: "12px" }}
          margin={{ top: 20, right: 20, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="1 1" />
          <XAxis
            dataKey="size_m2"
            type="number"
            name="Size"
            unit="m²"
            xAxisId="x"
          />
          <YAxis
            dataKey="price_$"
            type="number"
            name="Price"
            unit="$"
            yAxisId="y"
          />
          <ZAxis
            dataKey="price_in_m2"
            type="number"
            name="Price per m²"
            unit="$"
          />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Legend />
          <Scatter
            name="Properties"
            data={chartData}
            fill="#8884d8"
            xAxisId="x"
            yAxisId="y"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </>
  );
};

export default ScatterComp;
