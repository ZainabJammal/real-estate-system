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
import {
  useAggValues,
  useAllLists,
  useHotAreas,
  useProvList,
} from "../../Functions/apiLogic";
import { LuLoaderCircle } from "react-icons/lu";
import MapComponent from "../../Components/Map/MapComponent";

function Dashboard() {
  const {
    data: province,
    error: error_prov,
    isLoading: isLoading_prov,
  } = useProvList();

  const {
    data: all,
    error: error_all,
    isLoading: isLoading_all,
  } = useAllLists();

  const {
    data: agg_values,
    error: error_agg,
    isLoading: isLoading_agg,
  } = useAggValues();

  const {
    data: areas,
    error: error_areas,
    isLoading: isLoading_areas,
  } = useHotAreas();

  const max_listing_count = all?.reduce(
    ([max, location], district) =>
      district.listings_count > max
        ? [district.listings_count, district.district]
        : [max, location],
    [0, ""]
  );

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
              <MainCard data={agg_values} isLoading={isLoading_agg} />
            </div>
            <div className="card">
              <MainCard
                data={agg_values}
                isLoading={isLoading_agg}
                max={true}
              />
            </div>
            <div className="card">
              <MainCard
                data={agg_values}
                isLoading={isLoading_agg}
                min={true}
              />
            </div>
            <div className="card">
              <MainCard data={agg_values} isLoading={isLoading_agg} />
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
            {isLoading_prov && <LuLoaderCircle className="loader" size={30} />}
            {!isLoading_prov && (
              <Custom
                title="Prices/Provinces in $"
                desc={
                  "This chart represent the average, max and min prices of estates per province"
                }
                Component={BarChart}
                data={province}
                isLoading={isLoading_prov}
              />
            )}

            {isLoading_all && <LuLoaderCircle className="loader" size={30} />}
            {!isLoading_all && (
              <Custom
                title="Prices/Districts in $"
                desc={
                  "This chart represent the average, max and min prices of estates per district"
                }
                Component={BarChart}
                data={all}
                isLoading={isLoading_all}
              />
            )}
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
