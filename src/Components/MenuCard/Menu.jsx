import React from "react";
import "./Menu.css";
import { FaHome } from "react-icons/fa";

function Menu({ isMinimized = false }) {
  return (
    <div className={`item-list ${isMinimized ? "minimized" : ""}`}>
      <div className="item">
        {isMinimized ? <FaHome size={"50%"} /> : "Home"}
      </div>
      <div className="item">Explore Estates</div>
      <div className="item">Ask AI</div>
      <div className="item">Contact Agent</div>
    </div>
  );
}

export default Menu;
