<<<<<<< HEAD
import React from "react";
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import React, { useEffect, useState } from "react";
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
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
<<<<<<< HEAD

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
=======
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
<<<<<<< HEAD
            {!isMinimized && <span className="menu-text">{path.name}</span>}
          </li>
        </Link>
      ))}
    </ul>
=======
import React from "react";
=======
import React, { useState } from "react";
>>>>>>> 0ec06ac (5 - Added New Pages (Explore Estates, Ask AI, Contact Agent) and added their corresponding styles)
=======
import React, { useEffect, useState } from "react";
>>>>>>> b2365ee ((Add): added getPages.js and organized the menu component on the front-end)
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
<<<<<<< HEAD
    <div className={`item-list ${isMinimized ? "minimized" : ""}`}>
      <div className="item">
        {isMinimized ? <FaHome size={"50%"} /> : "Home"}
      </div>
      <div className="item">Explore Estates</div>
      <div className="item">Ask AI</div>
      <div className="item">Contact Agent</div>
    </div>
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
=======
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
            {path.icon}
=======
>>>>>>> ffe7405 ((create): created backend in Quart and Hypercorn to handle server-side logic and API database queries)
            {!isMinimized && <span className="menu-text">{path.name}</span>}
          </li>
        </Link>
      ))}
    </ul>
>>>>>>> 45d5ffb (Did some CSS modifications in the Sidebar.css)
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
  );
}

export default Menu;
