import React, { useEffect, useState } from "react";
import { CgAlignCenter } from "react-icons/cg";
import { PieChart, Pie, ResponsiveContainer } from "recharts";
import "./PieChartComponent.css";

function PieChartComponent() {
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener("resize", handleResize);

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const data = [
    { name: "Coast", value: 60 },
    { name: "Countryside", value: 10 },
    { name: "Mountains", value: 15 },
    { name: "Urban", value: 13 },
    { name: "City", value: 55 },
    { name: "Poor Neighborhoods", value: 35 },
    { name: "Hills", value: 22 },
  ];

  return (
    <div className="piechart-content">
      <ResponsiveContainer
        // width={windowWidth < 650 ? 350 : 600}
        // height={windowWidth < 650 ? 233 : 400}
        width={"100%"}
        height={"100%"}
      >
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={windowWidth < 500 ? 70 : 110}
            fill="#8884d8"
          />
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius={windowWidth < 500 ? 70 : 110}
            outerRadius={windowWidth < 500 ? 90 : 130}
            fill="#84d888"
            label
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

export default PieChartComponent;
