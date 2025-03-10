import React, { useEffect, useState } from "react";
import { CgAlignCenter } from "react-icons/cg";
import { PieChart, Pie, ResponsiveContainer, Cell } from "recharts";
import "./PieChartComponent.css";

function PieChartComponent({ data = null }) {
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener("resize", handleResize);

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // const data = [
  //   { name: "Coast", value: 60 },
  //   { name: "Countryside", value: 10 },
  //   { name: "Mountains", value: 15 },
  //   { name: "Urban", value: 13 },
  //   { name: "City", value: 55 },
  //   { name: "Poor Neighborhoods", value: 35 },
  //   { name: "Hills", value: 22 },
  // ];

  if (!data || data.length === 0) return <p>No data available</p>;

  // Extracting only the necessary fields for the pie chart
  const chartData = data?.map((area) => ({
    district: area.district, // or district if available
    listings: area.listings_count,
  }));

  const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7f50", "#0088FE"]; // Different colors for areas

  return (
    <>
      <ResponsiveContainer
        // width={windowWidth < 650 ? 350 : 600}
        // height={windowWidth < 650 ? 233 : 400}
        width={"100%"}
        height={300}
      >
        <PieChart>
          <Pie
            data={chartData}
            dataKey="listings"
            nameKey="district"
            cx="50%"
            cy="50%"
            outerRadius={windowWidth < 500 ? 50 : 70}
            fill="#8884d8"
          >
            {chartData.map((entry, index) => (
              <Cell key={index} fill={colors[index % colors.length]} />
            ))}
          </Pie>
          <Pie
            data={chartData}
            dataKey="listings"
            nameKey="district"
            cx="50%"
            cy="50%"
            innerRadius={windowWidth < 500 ? 50 : 70}
            outerRadius={windowWidth < 500 ? 70 : 90}
            fill="#84d888"
            label
          />
        </PieChart>
      </ResponsiveContainer>
    </>
  );
}

export default PieChartComponent;
