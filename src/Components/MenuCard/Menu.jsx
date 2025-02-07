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
    <div className="section">
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
    </div>
  );
}

export default Menu;
