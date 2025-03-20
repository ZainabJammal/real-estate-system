import React, { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Funnel,
  FunnelChart,
  LabelList,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const ListsByTypes = ({ data }) => {
  const [chartData, setChartData] = useState([]);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener("resize", handleResize);

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    if (data) {
      console.log(data);
      setChartData(
        data?.map((prop) => ({
          type: prop?.name,
          lists_count: prop?.value,
        }))
      );
      console.log(chartData);
    }
  }, [data]);

  const colors = [
    "#8884d8",
    "#82ca9d",
    "#ffc658",
    "#ff7f50",
    "#225788",
    "#FF2255",
    "#7810FF",
  ];

  return (
    <>
      <ResponsiveContainer width={"100%"} height={380}>
        {/* <FunnelChart width={730} height={200}>
          <Tooltip />
          <Funnel
            dataKey="lists_count"
            data={chartData}
            isAnimationActive
            stroke="white"
          >
            <LabelList
              position="right"
              fill="#000"
              stroke="none"
              dataKey="type"
            />
          </Funnel>
        </FunnelChart> */}

        <PieChart>
          <Pie
            data={chartData}
            dataKey="lists_count"
            nameKey="type"
            cx="50%"
            cy="50%"
            outerRadius={windowWidth < 500 ? 50 : 110}
            fill="#8884d8"
            label={({ type, lists_count }) => `${type}: ${lists_count}`}
          >
            {chartData.map((entry, index) => (
              <Cell key={index} fill={colors[(index * 5) % colors.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </>
  );
};

export default ListsByTypes;
