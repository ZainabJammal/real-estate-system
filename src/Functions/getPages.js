import { FaCloud, FaHome, FaList, FaPhoneAlt } from "react-icons/fa";

const menu_paths = [
  {
    name: "Dashboard",
    path: "/",
    icon: FaHome,
  },
  {
    name: "Transactions",
    path: "/transactions",
    icon: FaList,
  },
  {
    name: "Ask AI",
    path: "/ask_ai",
    icon: FaCloud,
  },
  {
    name: "Contact Agent",
    path: "/contact_agent",
    icon: FaPhoneAlt,
  },
];

export { menu_paths };
