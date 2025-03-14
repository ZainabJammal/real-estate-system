import React, { useEffect, useState } from "react";
import "./MainCard.css";
import { FaHome } from "react-icons/fa";
import { LuLoaderCircle } from "react-icons/lu";

const MainCard = ({
  data,
  isLoading,
  price = false,
  max = false,
  min = false,
}) => {
  const [values, setValues] = useState({ title: "", region: "", num: 0 });

  useEffect(() => {
    if (max && !min) {
      console.log("Setting the Max values");
      setValues((prev) => ({
        ...prev,
        title: data?.max,
        region: data?.region_max,
        num: data?.max_num,
      }));
      console.log("Values set, ", values);
    } else if (!max && !min) {
      setValues((prev) => ({
        ...prev,
        title: data?.sum,
        region: data?.region_all,
        num: data?.sum_num,
      }));
    } else {
      setValues((prev) => ({
        ...prev,
        title: data?.min,
        region: data?.region_min,
        num: data?.min_num,
      }));
    }
  }, [max, min, data]);

  return (
    <div className="main-card">
      {isLoading && <LuLoaderCircle className="loader" size={32} />}
      {!isLoading && (
        <>
          <div className="main-card-title">
            <h1>{values.title ? values.title : "Title"}</h1>
            <span>
              {values.num ? values.num : 0}
              {price ? "$" : ""}
            </span>
          </div>
          <div className="region">
            {values.region ? values.region : "Region"}
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
