import React, { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

const PriceM2 = ({ data }) => {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (data) {
      setChartData(
        data?.map((prop) => ({
          city: prop?.city,
          province: prop?.province,
          price_in_m2: (prop?.price_$ / prop?.size_m2).toFixed(2),
          type: prop?.type,
        }))
      );
    }
  }, [data]);

  // Calculate the max value of price_in_m2 dynamically
  const maxPriceInM2 = Math.max(
    ...chartData.map((item) => parseFloat(item.price_in_m2))
  );

  return (
    <>
      <ResponsiveContainer width={"100%"} height={380}>
        <BarChart
          style={{ fontSize: "12px" }}
          data={chartData}
          margin={{ top: 20, right: 20, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="1 1" />
          <YAxis domain={[0, maxPriceInM2]} />
          <XAxis dataKey="city" />
          <Tooltip />
          <Bar dataKey="price_in_m2" fill="#ff8712" />
          <Legend />
        </BarChart>
      </ResponsiveContainer>
    </>
  );
};

export default PriceM2;
