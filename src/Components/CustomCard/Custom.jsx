import React, { useEffect, useState } from "react";
import "./Custom.css";

<<<<<<< HEAD
function Custom({ title, desc, Component, no_inflate = false }) {
  return (
    <div className={no_inflate ? "card-layout no-inflate" : "card-layout"}>
=======
function Custom({ title, desc, Component }) {
  return (
    <div className="card-layout">
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
      <div className="card-content">
        <div className="card-title">
          <h3>{title ? title : "Title"}</h3>
          <p>{desc ? desc : "This card describes..."}</p>
        </div>
<<<<<<< HEAD
        <div className="card-main">
          {Component ? (
            <Component />
          ) : (
            <img src="../src/images/2.png" alt="chart.png" />
          )}
        </div>
=======
        {Component ? (
          <Component />
        ) : (
          <img src="../src/images/2.png" alt="chart.png" />
        )}
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
      </div>
    </div>
  );
}

export default Custom;
