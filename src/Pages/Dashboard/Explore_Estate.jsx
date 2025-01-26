<<<<<<< HEAD
import React, { useState } from "react";
import "./Page_Layout.css";
import { useQuery } from "@tanstack/react-query";

export const Explore_Estate = () => {
  const fetchData = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/user", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });
      if (!res.ok) {
        const error = await res.json();
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

=======
import React from "react";
import "./Page_Layout.css";

export const Explore_Estate = () => {
>>>>>>> 0ec06ac (5 - Added New Pages (Explore Estates, Ask AI, Contact Agent) and added their corresponding styles)
  return (
    <div className="dashboard-layout">
      <div className="dashboard-content">
        <div className="title">
          <h1>Explore Estate</h1>
        </div>
<<<<<<< HEAD
        <div className="dashboard-components">
          {isLoading && <p>Loading data...</p>}
          {error && <p>{error.message}</p>}
          {console.log(data)}
          {!isLoading &&
            data.map((user, id) => (
              <div key={user.id}>
                <p>{user.username}</p>
                <p>{user.password}</p>
              </div>
            ))}
        </div>
=======
        <div className="dashboard-components"></div>
>>>>>>> 0ec06ac (5 - Added New Pages (Explore Estates, Ask AI, Contact Agent) and added their corresponding styles)
      </div>
    </div>
  );
};

export default Explore_Estate;
