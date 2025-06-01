<<<<<<< HEAD
import { FaCloud, FaHome, FaChartLine, FaTable, FaChromecast,  FaSnapchat } from "react-icons/fa";
=======
import { FaCloud, FaHome, FaChartLine, FaTable } from "react-icons/fa";

>>>>>>> main

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
    name: "Chatbot",
=======
>>>>>>> main
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
