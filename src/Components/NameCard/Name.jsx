import React from "react";
import "./Name.css";

function Name({ isMinimized = false }) {
  return (
    <div className={`name ${isMinimized ? "minimized" : ""}`}>
      <img src="../src/images/1.jpg" alt="avatar.png" />
      {isMinimized ? (
        ""
      ) : (
        <h1>
          <i>Real Estate</i>
        </h1>
      )}
    </div>
  );
}

export default Name;
