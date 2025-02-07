import React, { useState } from "react";
import "./Page_Layout.css";
import { useQuery } from "@tanstack/react-query";
import { LuLoaderCircle } from "react-icons/lu";

export const Explore_Estate = () => {
  const fetchData = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/transactions", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });
      if (!res.ok) {
        const { error } = await res.json();
        throw new Error(error.message || "Something went wrong");
      }
      return res.json();
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
          <h1>Explore Estate</h1>
        </div>
        <div className="dashboard-components">
          {isLoading && <LuLoaderCircle className="loader center" size={25} />}
          {error && <p>{error.message}</p>}
          {console.log(data)}
          {!isLoading &&
            data?.map((user, id) => (
              <div key={user.id}>
                <p>{user.username}</p>
                <p>{user.password}</p>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};

export default Explore_Estate;
