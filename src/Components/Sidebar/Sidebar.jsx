import React from "react";
import "./Sidebar.css";
import Name from "../NameCard/Name";
import Menu from "../MenuCard/Menu";
import { FaArrowAltCircleLeft } from "react-icons/fa";

function Sidebar() {
  return (
    <div className="sidebar">
      <div className="navmenu">
        <Name />
        <Menu />
      </div>
      <div className="logout-container">
        <div className="logout">
          <FaArrowAltCircleLeft />
        </div>
      </div>
    </div>
  );
}

export default Sidebar;
