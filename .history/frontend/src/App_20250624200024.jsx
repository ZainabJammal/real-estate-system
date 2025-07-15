import { useState } from "react";
import { BrowserRouter as BR, Router, Route, Routes } from "react-router-dom";
import "./App.css";
import Sidebar from "./Components/Sidebar/Sidebar";
import Dashboard from "./Pages/Dashboard/Dashboard";
import Transactions from "./Pages/Dashboard/Transactions";
import Tables from "./Pages/Dashboard/Tables";
// import Ask_AI from "./Pages/Dashboard/Ask_AI";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ChatAssistant from "./Pages/ChatAssistant";
import TransactionsForecasting from "./Pages/TransactionsForecasting";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BR>
        <div className="app">
          <Sidebar />

          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/transactions" element={<Transactions />} />
            {/* <Route path="/ask_ai" element={<Ask_AI />} /> */}
            <Route path="/transactions_forecasting" element={<TransactionsForecasting />} />
             <Route path="/transactions_forecasting" element={<TransactionsForecasting />} />
            <Route path="/chatbot" element={<ChatAssistant />} />
            <Route path="/tables" element={<Tables />} />
          </Routes>
        </div>
      </BR>
    </QueryClientProvider>
  );
}

export default App;
