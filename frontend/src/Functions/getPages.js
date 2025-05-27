import { FaCloud, FaHome, FaChartLine, FaTable, FaChromecast, FaSnapchat } from "react-icons/fa";

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
    name: "Ask AI",
    path: "/ask_ai",
    icon: FaCloud,
  },
  {
    name: "Chatbot",
    path: "/chatbot",
    icon: FaChromecast,
  },
  {
    name: "Tables",
    path: "/tables",
    icon: FaTable,
  },
];

export { menu_paths };
