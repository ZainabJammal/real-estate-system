import React, { useEffect, useState } from "react";
import "./Custom.css";

function Custom({ title, desc, Component }) {
  return (
    <div className="card-layout">
      <div className="card-content">
        <div className="card-title">
          <h3>{title ? title : "Title"}</h3>
          <p>{desc ? desc : "This card describes..."}</p>
        </div>
        {Component ? (
          <Component />
        ) : (
          <img src="../src/images/2.png" alt="chart.png" />
        )}
      </div>
    </div>
  );
}

export default Custom;
