import React, { useEffect, useState } from "react";
import {
  CartesianGrid,
  Label,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const LineC = ({ data }) => {
  const [chartData, setChartData] = useState([
    { date: "1", transaction_value: "10" },
  ]);
  useEffect(() => {
    if (data) {
      setChartData(
        data?.map((prop) => ({
          city: prop?.city,
          price_$: prop?.price_$,
          type: prop?.type,
        }))
      );
    }

    const uniqueData = data?.reduce((acc, property) => {
      // Check if the city already exists in the accumulator
      if (!acc[property.city] || property.size > acc[property.city].size) {
        acc[property.city] = property; // Keep the larger property
      }
      return acc;
    }, {});

    if (uniqueData) {
      // Convert object values back to an array
      setChartData(Object.values(uniqueData));
    }
  }, [data]);

  console.log(chartData);

  return (
    <>
      <ResponsiveContainer width={"100%"} height={420}>
        <LineChart
          data={chartData.sort((a, b) => a.price_$ - b.price_$)}
          margin={{ top: 30, right: 50, left: 20, bottom: 20 }}
          style={{ fontSize: ".8rem" }}
        >
          <CartesianGrid strokeDasharray="1 1" />
          <XAxis dataKey="city">
            <Label value="Cities" offset={-5} position="insideBottom" />
          </XAxis>
          <YAxis>
            <Label
              value="Price per (m²)"
              offset={360}
              position="insideBottom"
            />
          </YAxis>
          <Tooltip />
          {/* <Legend /> */}

          <Line type="monotone" dataKey="price_$" stroke="#fe4533" />
          {/* <Line type="monotone" dataKey="Agent2" stroke="#da829d" /> */}
        </LineChart>
      </ResponsiveContainer>
    </>
  );
};

export default LineC;
