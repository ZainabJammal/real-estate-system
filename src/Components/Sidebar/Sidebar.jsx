import React, { useState } from "react";
import "./Sidebar.css";
import Name from "../NameCard/Name";
import Menu from "../MenuCard/Menu";
import { FaArrowAltCircleLeft } from "react-icons/fa";

function Sidebar() {
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
      </div>
    </div>
  );
}

export default Sidebar;
