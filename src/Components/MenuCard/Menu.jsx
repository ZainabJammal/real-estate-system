import React from "react";
import "./Menu.css";
import {
  FaChartArea,
  FaCloud,
  FaFileContract,
  FaHome,
  FaLightbulb,
  FaList,
  FaPage4,
  FaPaperPlane,
  FaPhoneAlt,
} from "react-icons/fa";
import { FaPhone } from "react-icons/fa6";

function Menu({ isMinimized = false }) {
  return (
    <ul className={`item-list ${isMinimized ? "minimized" : ""}`}>
      <li
        className={`item ${isMinimized ? "minimized" : ""}`}
        aria-label="Home"
      >
        <FaHome size={"20px"} className="menu-icon" />
        {!isMinimized && <span className="menu-text">Home</span>}
      </li>
      <li
        className={`item ${isMinimized ? "minimized" : ""}`}
        aria-label="Explore Estates"
      >
        <FaList size={"20px"} className="menu-icon" />
        {!isMinimized && <span className="menu-text">Explore Estates</span>}
      </li>
      <li
        className={`item ${isMinimized ? "minimized" : ""}`}
        aria-label="Ask AI"
      >
        <FaCloud size={"20px"} className="menu-icon" />
        {!isMinimized && <span className="menu-text">Ask AI</span>}
      </li>
      <li
        className={`item ${isMinimized ? "minimized" : ""}`}
        aria-label="Contact Agent"
      >
        <FaPhoneAlt size={"20px"} className="menu-icon" />
        {!isMinimized && <span className="menu-text">Contact Agent</span>}
      </li>
    </ul>
  );
}

export default Menu;
