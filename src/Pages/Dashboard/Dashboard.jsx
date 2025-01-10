import React from "react";
import "./Dashboard.css";
import Name from "../../Components/NameCard/Name";

function Dashboard() {
  return (
    <div className="dashboard-layout">
      <div className="dashboard-content">
        <div className="title">
          <h1>Dashboard</h1>
        </div>
        <div className="dashboard-components"></div>
      </div>
    </div>
  );
}

export default Dashboard;
