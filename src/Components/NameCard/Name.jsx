import React from "react";
import "./Name.css";

<<<<<<< HEAD
<<<<<<< HEAD
function Name({ isMinimized = false }) {
  return (
    <div className={`name ${isMinimized ? "minimized" : ""}`}>
      <img src="../src/images/1.jpg" alt="avatar.png" />
      {isMinimized ? (
        ""
      ) : (
        <h1>
          <i>Real Estate</i>
        </h1>
      )}
=======
function Name() {
  return (
    <div className="name">
<<<<<<< HEAD
      <img src="../src/images/1.png" alt="avatar.png" />
      <h1>Mostafa Dawi!</h1>
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
=======
      <img src="../src/images/1.jpg" alt="avatar.png" />
      <h1>
        <i>Real Estate</i>
      </h1>
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
=======
function Name({ isMinimized = false }) {
  return (
    <div className={`name ${isMinimized ? "minimized" : ""}`}>
      <img src="../src/images/1.jpg" alt="avatar.png" />
      {isMinimized ? (
        ""
      ) : (
        <h1>
          <i>Real Estate</i>
        </h1>
      )}
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
    </div>
  );
}

export default Name;
