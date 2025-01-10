import { useState } from "react";
<<<<<<< HEAD
import { BrowserRouter as BR, Router, Route, Routes } from "react-router-dom";
import "./App.css";
import Sidebar from "./Components/Sidebar/Sidebar";
import Dashboard from "./Pages/Dashboard/Dashboard";
import Explore_Estate from "./Pages/Dashboard/Explore_Estate";
import Contact_Agent from "./Pages/Dashboard/Contact_Agent";
import Ask_AI from "./Pages/Dashboard/Ask_AI";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BR>
        <div className="app">
          <Sidebar />

          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/explore_estates" element={<Explore_Estate />} />
            <Route path="/ask_ai" element={<Ask_AI />} />
            <Route path="/contact_agent" element={<Contact_Agent />} />
          </Routes>
        </div>
      </BR>
    </QueryClientProvider>
=======
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
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
  );
}

export default App;
