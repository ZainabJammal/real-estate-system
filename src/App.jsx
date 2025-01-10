import { useState } from "react";
import { BrowserRouter as BR, Router, Route } from "react-router-dom";
import "./App.css";
import Sidebar from "./Components/Sidebar/Sidebar";
import Dashboard from "./Pages/Dashboard/Dashboard";

function App() {
  return (
    <div className="app">
      <Sidebar />
      <Dashboard />
    </div>
  );
}

export default App;
