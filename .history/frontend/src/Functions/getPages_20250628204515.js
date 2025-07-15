import { FaCloud, FaHome, FaChartLine, FaTable } from "react-icons/fa";
import {  MdPriceCheck } from "react-icons/md";

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
    name: "Fair Price Estimator",
    path: "/price_estimat",
    icon: MdPriceCheck,
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
