import React, { useEffect, useState } from "react";
import "./Custom.css";

<<<<<<< HEAD
function Custom({ title, desc, Component, no_inflate = false }) {
  return (
    <div className={no_inflate ? "card-layout no-inflate" : "card-layout"}>
=======
<<<<<<< HEAD
<<<<<<< HEAD
function Custom({ title, desc, Component, no_inflate = false }) {
  return (
    <div className={no_inflate ? "card-layout no-inflate" : "card-layout"}>
=======
function Custom({ title, desc, Component }) {
  return (
    <div className="card-layout">
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
=======
function Custom({ title, desc, Component, no_inflate = false }) {
  return (
    <div className={no_inflate ? "card-layout no-inflate" : "card-layout"}>
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
      <div className="card-content">
        <div className="card-title">
          <h3>{title ? title : "Title"}</h3>
          <p>{desc ? desc : "This card describes..."}</p>
        </div>
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
        <div className="card-main">
          {Component ? (
            <Component />
          ) : (
            <img src="../src/images/2.png" alt="chart.png" />
          )}
        </div>
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
        {Component ? (
          <Component />
        ) : (
          <img src="../src/images/2.png" alt="chart.png" />
        )}
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
      </div>
    </div>
  );
}

export default Custom;
