import React from "react";
import "./Page_Layout.css";
import Custom from "../../Components/CustomCard/Custom";
import Table from "../../Components/Table/Table";
import PieChart from "../../Components/PieChart/PieChartComponent";
import LineChartComponent from "../../Components/LineChart/LineChartComponent";
import Name from "../../Components/NameCard/Name";
import { Bar } from "recharts";
import BarChart from "../../Components/BarChart/BarChartComponent";
import MainCard from "../../Components/MainCard/MainCard";
import { useQuery } from "@tanstack/react-query";
import { LuLoaderCircle } from "react-icons/lu";
import MapComponent from "../../Components/Map/MapComponent";

function Dashboard() {
  const fetchNumLists = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/list_num", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(res.error.message || "Something went wrong");
      }

      return data;
    } catch (error) {
      console.error("Cannot Fetch: ", error);
      throw error;
    }
  };

  const fetchMaxPrice = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/max_price", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(res.error.message || "Something went wrong");
      }

      return data;
    } catch (error) {
      console.error("Cannot Fetch: ", error);
      throw error;
    }
  };

  const fetchMinPrice = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/min_price", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(res.error.message || "Something went wrong");
      }

      return data;
    } catch (error) {
      console.error("Cannot Fetch: ", error);
      throw error;
    }
  };

  const fetchAllLists = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/all_lists", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(res.error.message || "Something went wrong");
      }

      return data;
    } catch (error) {
      console.error("Cannot Fetch: ", error);
      throw error;
    }
  };

  const fetchProvLists = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/provinces", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(res.error.message || "Something went wrong");
      }

      return data;
    } catch (error) {
      console.error("Cannot Fetch: ", error);
      throw error;
    }
  };

  const fetchHotAreas = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/hot_areas", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(res.error.message || "Something went wrong");
      }

      return data;
    } catch (error) {
      console.error("Cannot Fetch: ", error);
      throw error;
    }
  };

  const {
    data: province,
    error: error_prov,
    isLoading: isLoading_prov,
  } = useQuery({
    queryKey: ["myProvData"],
    queryFn: fetchProvLists,
  });

  const {
    data: all,
    error: error_all,
    isLoading: isLoading_all,
  } = useQuery({
    queryKey: ["myAllData"],
    queryFn: fetchAllLists,
  });

  const {
    data: sum,
    error: error_sum,
    isLoading: isLoading_sum,
  } = useQuery({
    queryKey: ["myListData"],
    queryFn: fetchNumLists,
  });

  const {
    data: max,
    error: error_max,
    isLoading: isLoading_max,
  } = useQuery({
    queryKey: ["myMaxData"],
    queryFn: fetchMaxPrice,
  });

  const {
    data: min,
    error: error_min,
    isLoading: isLoading_min,
  } = useQuery({
    queryKey: ["myMinData"],
    queryFn: fetchMinPrice,
  });

  const {
    data: areas,
    error: error_areas,
    isLoading: isLoading_areas,
  } = useQuery({
    queryKey: ["myHotAreasData"],
    queryFn: fetchHotAreas,
  });

  const max_listing_count = all?.reduce(
    ([max, location], district) =>
      district.listings_count > max
        ? [district.listings_count, district.district]
        : [max, location],
    [0, ""]
  );

  console.log(all);
  console.log(province);
  console.log(areas);

  return (
    <div className="dashboard-layout">
      <div className="dashboard-content">
        <div className="title">
          <h1>Dashboard</h1>
        </div>
        <div className="section">
          <h1>For Sale</h1>
          <div className="main-cards">
            <div className="card">
              <MainCard data={sum} isLoading={isLoading_sum} />
            </div>
            <div className="card">
              <MainCard data={max} isLoading={isLoading_max} price={true} />
            </div>
            <div className="card">
              <MainCard data={min} isLoading={isLoading_min} price={true} />
            </div>
            <div className="card">
              <MainCard data={sum} />
            </div>
          </div>
        </div>
        <div className="section">
          <h1>Stats</h1>
          <div className="dashboard-components grid-3">
            {isLoading_all && <LuLoaderCircle className="loader" size={30} />}
            {!isLoading_all && (
              <Custom
                title="Number of Highly Demanded Estates per District"
                desc={`Hottest area in Lebanon would be ${
                  !isLoading_all && max_listing_count[1]
                }, having ${
                  !isLoading_all && max_listing_count[0]
                } estates available`}
                Component={PieChart}
                data={all}
              />
            )}

            <Custom
              title="Prices/Provinces in $"
              desc={
                "This chart represent the average, max and min prices of estates per province"
              }
              Component={BarChart}
              data={province}
            />

            <Custom
              title="Prices/Districts in $"
              desc={
                "This chart represent the average, max and min prices of estates per district"
              }
              Component={BarChart}
              data={all}
            />
          </div>
        </div>
        <div className="section">
          <h1>Map</h1>
          <div className="dashboard-components grid-1">
            {isLoading_areas && <LuLoaderCircle className="loader" size={30} />}
            {!isLoading_areas && (
              <Custom
                title="Hotest areas (Map)"
                desc={"Coming soon..."}
                Component={MapComponent}
                data={areas}
              />
            )}
          </div>
        </div>
        <div className="title">
          <h1>Tables</h1>
        </div>
        <div className="section">
          <div className="dashboard-components grid-1">
            <Custom
              title="District Estates"
              desc="Table of availabe estates per districts"
              Component={Table}
              data={all}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
