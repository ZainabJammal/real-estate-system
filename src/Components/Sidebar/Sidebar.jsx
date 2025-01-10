<<<<<<< HEAD
import React, { useState } from "react";
=======
import React from "react";
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
import "./Sidebar.css";
import Name from "../NameCard/Name";
import Menu from "../MenuCard/Menu";
import { FaArrowAltCircleLeft } from "react-icons/fa";

function Sidebar() {
<<<<<<< HEAD
  const [isMinimized, setMinimized] = useState(false);
  const toggleSidebar = () => {
    setMinimized((prev) => !prev);
  };
  return (
    <div className={`sidebar ${isMinimized ? "minimized" : ""}`}>
      <div className="navmenu">
        <Name isMinimized={isMinimized} />
        <Menu isMinimized={isMinimized} />
      </div>
      <div className="logout-container">
        <div className={`logout ${isMinimized ? "minimized" : ""}`}>
          <FaArrowAltCircleLeft onClick={toggleSidebar} />
        </div>
=======
  return (
    <div className="sidebar">
      <div className="navmenu">
        <Name />
        <Menu />
      </div>
      <div className="logout">
        <FaArrowAltCircleLeft />
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
      </div>
    </div>
  );
}

export default Sidebar;
