import React from "react";
import "./Name.css";

function Name({ isMinimized = false }) {
  return (
    <ul className={`name ${isMinimized ? "minimized" : ""}`}>
      <li>
        <img src="../src/images/1.jpg" alt="avatar.png" />
      </li>
      {isMinimized ? (
        ""
      ) : (
        <li>
          <h1>
            <i>Real Estate</i>
          </h1>
        </li>
      )}
    </ul>
  );
}

export default Name;
