import React, { useState } from "react";
import "./Page_Layout.css";
import { useQuery } from "@tanstack/react-query";
import { LuLoaderCircle } from "react-icons/lu";
import "../../Components/Table/Table.css";
import Custom from "../../Components/CustomCard/Custom";
import LineChartComponent from "../../Components/LineChart/LineChartComponent";
import Table from "../../Components/Table/Table";

export const Transactions = () => {
  const fetchData = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/transactions", {
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

  const { data, error, isLoading } = useQuery({
    queryKey: ["myData"],
    queryFn: fetchData,
  });

  console.log(data);

  return (
    <div className="dashboard-layout">
      <div className="dashboard-content">
        <div className="title">
          <h1>Transactions</h1>
          {isLoading && <LuLoaderCircle className="loader center" size={25} />}
          {error && <p>{error.message}</p>}
          {console.log(data)}
        </div>
        <div className="section">
          <div className="dashboard-components grid-1">
            <Custom
              title={"Transactions made in real estates in Lebanon (2011-2016)"}
              desc={
                "This table represents the distribution of transactions per month/year across different regions"
              }
              Component={Table}
              data={data}
            />

            <Custom
              title={"Transactions per year"}
              desc={"Variation of transactions as the years pass"}
              Component={LineChartComponent}
              data={data}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Transactions;
