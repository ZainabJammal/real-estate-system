import { FaCloud, FaHome, FaChartLine, FaTable } from "react-icons/fa";
import {  MdPriceCheck } from "react-icons/md";
import { MdCompareArrows } from "react-icons/md";
import { TbChartArrows } from "react-icons/tb";




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
    icon: FaChartLine,
  },
  {
    name: "Fair Price Estimator",
    path: "/price_estimator",
    icon: MdPriceCheck,
  },
   {
    name: "Comparative Market Analysis",
    path: "/market_comparison",
    icon: MdCompareArrows,
  },
  {
    name: "Ask AI",
    path: "/chatbot",
    icon: FaCloud,
  },
   {
    name: "Market Scenario Simulator",
    path: "/market_simulator",
    icon: TbChartArrows,
  },

  {
    name: "Tables",
    path: "/tables",
    icon: FaTable,
  },
];

export { menu_paths };
