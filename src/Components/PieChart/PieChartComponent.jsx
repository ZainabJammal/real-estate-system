import React from "react";
import { PieChart, Pie } from "recharts";

function PieChartComponent() {
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
    </div>
  );
}

export default PieChartComponent;
