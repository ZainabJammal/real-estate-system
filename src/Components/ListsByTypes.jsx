import React, { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Funnel,
  FunnelChart,
  LabelList,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const ListsByTypes = ({ data }) => {
  const [chartData, setChartData] = useState([]);

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

  return (
    <>
      <ResponsiveContainer width={"100%"} height={380}>
        <FunnelChart width={730} height={200}>
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
        </FunnelChart>
      </ResponsiveContainer>
    </>
  );
};

export default ListsByTypes;
