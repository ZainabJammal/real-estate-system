<<<<<<< HEAD
import React, { useState } from "react";
=======
<<<<<<< HEAD
<<<<<<< HEAD
import React, { useState } from "react";
=======
import React from "react";
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
=======
import React, { useState } from "react";
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
import "./Sidebar.css";
import Name from "../NameCard/Name";
import Menu from "../MenuCard/Menu";
import { FaArrowAltCircleLeft } from "react-icons/fa";

function Sidebar() {
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
  const [isMinimized, setMinimized] = useState(false);
  const toggleSidebar = () => {
    setMinimized((prev) => !prev);
  };
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
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
<<<<<<< HEAD
=======
=======
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
  return (
    <div className={`sidebar ${isMinimized ? "minimized" : ""}`}>
      <div className="navmenu">
        <Name isMinimized={isMinimized} />
        <Menu isMinimized={isMinimized} />
      </div>
<<<<<<< HEAD
      <div className="logout">
        <FaArrowAltCircleLeft />
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
=======
      <div className="logout-container">
        <div className={`logout ${isMinimized ? "minimized" : ""}`}>
          <FaArrowAltCircleLeft onClick={toggleSidebar} />
        </div>
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
      </div>
    </div>
  );
}

export default Sidebar;
