import React from "react";
import "./Page_Layout.css";
import Custom from "../../Components/CustomCard/Custom";
import Table from "../../Components/Table/Table";
import PieChart from "../../Components/PieChart/PieChartComponent";
import LineChartComponent from "../../Components/LineChart/LineChartComponent";
import Name from "../../Components/NameCard/Name";
import { Bar } from "recharts";
import BarChart from "../../Components/BarChart/BarChartComponent";
import MainCard from "../../Components/MainCard/MainCard";
import { useQuery } from "@tanstack/react-query";

function Dashboard() {
  const fetchNumLists = async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/list_num", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(res.error.message || "Something went wrong");
      }

      return data;
    } catch (error) {
      console.error("Cannot Fetch: ", error);
      throw error;
    }
  };

  const fetchMaxPrice = async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/max_price", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(res.error.message || "Something went wrong");
      }

      return data;
    } catch (error) {
      console.error("Cannot Fetch: ", error);
      throw error;
    }
  };

  const fetchMinPrice = async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/min_price", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(res.error.message || "Something went wrong");
      }

      return data;
    } catch (error) {
      console.error("Cannot Fetch: ", error);
      throw error;
    }
  };

  const {
    data: sum,
    error: error_sum,
    isLoading: isLoading_sum,
  } = useQuery({
    queryKey: ["myListData"],
    queryFn: fetchNumLists,
  });

  const {
    data: max,
    error: error_max,
    isLoading: isLoading_max,
  } = useQuery({
    queryKey: ["myMaxData"],
    queryFn: fetchMaxPrice,
  });

  const {
    data: min,
    error: error_min,
    isLoading: isLoading_min,
  } = useQuery({
    queryKey: ["myMinData"],
    queryFn: fetchMinPrice,
  });

  return (
    <div className="dashboard-layout">
      <div className="dashboard-content">
        <div className="title">
          <h1>Dashboard</h1>
        </div>
        <div className="section">
          <h1>For Sale</h1>
          <div className="main-cards">
            <div className="card">
              <MainCard data={sum} isLoading={isLoading_sum} />
            </div>
            <div className="card">
              <MainCard data={max} isLoading={isLoading_max} price={true} />
            </div>
            <div className="card">
              <MainCard data={min} isLoading={isLoading_min} price={true} />
            </div>
            <div className="card">
              <MainCard data={sum} />
            </div>
          </div>
        </div>
        <div className="section">
          <h1>Stats</h1>
          <div className="dashboard-components">
            <Custom
              title="Sales per Month"
              desc="This line chart shows the increasing amount of sales (in $) per each month"
              Component={LineChartComponent}
            />

            <Custom
              title="Percentage of Highly Demanded Estates"
              desc="This chart represents the number of highly demanded estates according to locations"
              Component={PieChart}
            />

            <Custom
              title="Sales per Month"
              desc="This line chart shows the increasing amount of sales (in $) per each month"
              Component={LineChartComponent}
            />
            <Custom Component={BarChart} />
          </div>
        </div>
        <div className="title">
          <h1>Tables</h1>
        </div>
        <div className="dashboard-components">
          <Custom
            type="table"
            title="Table of availabe estates"
            Component={Table}
          />
          <Custom />
          <Custom />
          <Custom
            title="Sales per Month"
            desc="This line chart shows the increasing amount of sales (in $) per each month"
            Component={LineChartComponent}
          />
          <Custom />
          <Custom />

          <Custom
            type="table"
            title="Table of availabe estates"
            Component={Table}
            no_inflate
          />
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
