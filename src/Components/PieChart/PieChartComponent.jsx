<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
import React, { useEffect, useState } from "react";
import { CgAlignCenter } from "react-icons/cg";
import { PieChart, Pie, ResponsiveContainer } from "recharts";
import "./PieChartComponent.css";
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30

function PieChartComponent() {
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener("resize", handleResize);

    return () => window.removeEventListener("resize", handleResize);
  }, []);
<<<<<<< HEAD
=======
=======
import React from "react";
import { PieChart, Pie } from "recharts";

function PieChartComponent() {
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
=======

function PieChartComponent() {
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener("resize", handleResize);

    return () => window.removeEventListener("resize", handleResize);
  }, []);
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
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
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
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
            outerRadius={windowWidth < 500 ? 70 : 125}
            fill="#8884d8"
          />
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            innerRadius={windowWidth < 500 ? 70 : 125}
            outerRadius={windowWidth < 500 ? 90 : 160}
            fill="#84d888"
            label
          />
        </PieChart>
      </ResponsiveContainer>
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
      <PieChart width={550} height={350}>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={115}
          fill="#8884d8"
        />
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          innerRadius={120}
          outerRadius={140}
          fill="#84d888"
          label
        />
      </PieChart>
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
    </div>
  );
}

export default PieChartComponent;
