import React from "react";
import "./MainCard.css";
import { FaHome } from "react-icons/fa";
import { LuLoaderCircle } from "react-icons/lu";

const MainCard = ({ data, isLoading, price = false }) => {
  return (
    <div className="main-card">
      {isLoading && <LuLoaderCircle className="loader" size={32} />}
      {!isLoading && (
        <>
          <div className="main-card-title">
            <h1>{data?.title || "title"}</h1>
            <span>
              {data?.num || 0}
              {price ? "$" : ""}
            </span>
          </div>
          <div className="region">
            {data?.region}
            <div className="icon">
              <FaHome />
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default MainCard;
