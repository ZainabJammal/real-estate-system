
import { FaCloud, FaHome, FaChartLine, FaTable } from "react-icons/fa";


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
    name: "TimeSeries Forecasting",
    path: "/TimeSeries_forecasting",
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
