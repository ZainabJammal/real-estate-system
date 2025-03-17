import React, { useEffect, useState } from "react";
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
  useProperties,
  useProvList,
  useTypeNums,
} from "../../Functions/apiLogic";
import { LuLoaderCircle } from "react-icons/lu";
import MapComponent from "../../Components/Map/MapComponent";
import PriceM2 from "../../Components/PriceM2";
import ListsByTypes from "../../Components/ListsByTypes";

function Dashboard() {
  const [gridValue, setGridValue] = useState(0);
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

  const {
    data: properties,
    error: error_properties,
    isLoading: isLoading_properties,
  } = useProperties();

  const {
    data: lists_type,
    error: error_type,
    isLoading: isLoading_type,
  } = useTypeNums();

  const max_listing_count = all?.reduce(
    ([max, location], district) =>
      district.listings_count > max
        ? [district.listings_count, district.district]
        : [max, location],
    [0, ""]
  );

  const handleGridChange = () => {
    setGridValue(!gridValue);
  };

  return (
    <div className="dashboard-layout">
      <div className="dashboard-content">
        <div className="title">
          <h1>Dashboard</h1>
          <button onClick={() => handleGridChange()}>...</button>
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
              <MainCard
                data={agg_values}
                isLoading={isLoading_agg}
                tran={true}
              />
            </div>
          </div>
        </div>
        {/* FIRST ANALYSIS SCTION */}
        <div className="section">
          <h1>Stats</h1>
          <div
            className={`dashboard-components ${
              gridValue ? "grid-1" : "grid-2"
            }`}
          >
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
            {console.log(province)}

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
            {isLoading_properties && (
              <LuLoaderCircle className="loader" size={30} />
            )}
            {!isLoading_properties && (
              <Custom
                title="Prices/Squared Meter"
                desc={
                  "This chart shows the distribution of the prices per meter squared across different cities in Lebanon"
                }
                Component={PriceM2}
                data={properties}
                isLoading={isLoading_properties}
              />
            )}
          </div>
        </div>
        {/* SECOND ANALYSIS SECTION */}
        <div className="section">
          <div
            className={`dashboard-components ${
              gridValue ? "grid-1" : "grid-3"
            }`}
          >
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

            {isLoading_type && <LuLoaderCircle className="loader" size={30} />}
            {!isLoading_type && (
              <Custom
                title="Distribution by Types"
                desc={
                  "This chart shows the distribution of the prices per meter squared across different cities in Lebanon"
                }
                Component={ListsByTypes}
                data={lists_type}
                isLoading={isLoading_type}
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
