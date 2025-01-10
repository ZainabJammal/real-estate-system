import React from "react";
import "./Name.css";

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
      <img src="../src/images/1.png" alt="avatar.png" />
      <h1>Mostafa Dawi!</h1>
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
    </div>
  );
}

export default Name;
