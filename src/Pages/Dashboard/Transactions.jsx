import React, { useState } from "react";
import "./Page_Layout.css";
import { useQuery } from "@tanstack/react-query";
import { LuLoaderCircle } from "react-icons/lu";
import "../../Components/Table/Table.css";
import Custom from "../../Components/CustomCard/Custom";
import LineChartComponent from "../../Components/LineChart/LineChartComponent";

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
        const { error } = await res.json();
        throw new Error(error.message || "Something went wrong");
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
          <h1>Transactions</h1>
          <div className="dashboard-components">
            <div className="table-card">
              <table>
                <thead>
                  <th>Transaction ID</th>
                  <th>Transaction Value</th>
                  <th>Date</th>
                </thead>
                <tbody>
                  {!isLoading &&
                    data?.map((trans, id) => (
                      <tr key={trans.id}>
                        <td>{trans?.transaction_number}</td>
                        <td>{trans?.transaction_value}</td>
                        <td>{trans?.date}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
            <Custom
              title={"Transactions per year"}
              desc={"Variation of transactions as the years increase"}
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
