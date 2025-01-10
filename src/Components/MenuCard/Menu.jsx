<<<<<<< HEAD
import React, { useEffect, useState } from "react";
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
import { Link, useLocation } from "react-router-dom";
import { menu_paths } from "../../Functions/getPages";

function Menu({ isMinimized = false }) {
  const path = useLocation();
  const [activePage, setActivePage] = useState(null);

  function handleClick(path) {
    setActivePage(path);
  }

  useEffect(() => {
    setActivePage(path.pathname);
  }, [path]);

  return (
    <ul className={`item-list ${isMinimized ? "minimized" : ""}`}>
      {menu_paths.map((path, _idx) => (
        <Link
          to={path.path}
          key={_idx}
          style={{ textDecoration: "none", color: "inherit" }}
        >
          <li
            className={`item ${isMinimized ? "minimized" : ""} ${
              activePage === path.path ? "clicked" : ""
            }`}
            aria-label={path.name}
            onClick={() => handleClick(path.path)}
          >
            <path.icon size={"20px"} className="menu-icon" />
            {!isMinimized && <span className="menu-text">{path.name}</span>}
          </li>
        </Link>
      ))}
    </ul>
=======
import React from "react";
import "./Menu.css";

function Menu() {
  return (
    <div className="item-list">
      <div className="item">Home</div>
      <div className="item">Explore Estates</div>
      <div className="item">Ask AI</div>
      <div className="item">Contact Agent</div>
    </div>
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
  );
}

export default Menu;
