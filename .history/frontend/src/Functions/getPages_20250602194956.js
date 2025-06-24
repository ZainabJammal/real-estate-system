import { FaCloud, FaHome, FaChartLine, FaTable } from "react-icons/fa";
import {  MdTrendingUp } from "react-icons/md";

const menu_paths = [
  {
    name: "Dashboard",
    path: "/",
    icon: FaHome,
  },
  {
    name: "Transactions",
    path: "/transactions",
    icon: FaChartLine,
  },
  {
    name: "Transactions Forecasting",
    path: "/Transactions_Forecasting",
    icon: FaCloud,
  },
  {
    name: "Prices Forecasting",
    path: "/Transactions_Forecasting",
    icon: FaCloud,
  },
  {
    name: "Ask AI",
    path: "/chatbot",
    icon: FaCloud,
  },
  {
    name: "Tables",
    path: "/tables",
    icon: FaTable,
  },
];

export { menu_paths };
