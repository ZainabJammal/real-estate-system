import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, Tooltip, Circle } from "react-leaflet";
import "./MapComponent.css"; 
import arrowUpTrend from "../../assets/arrow-trend-up.png";
import arrowDownTrend from "../../assets/arrow-trend-down.png";
import NeutralTrend from "../../assets/neutral_trend.png";


const MapComponent = ({ data }) => {
  const initialData = data.province;
  const [selectedData, setSelectedData] = useState(initialData);
  const [selectedOption, setSelectedOption] = useState("Province");
  const [cityCircles, setCityCircles] = useState([]);
  // const [cityRanges, setCityRanges] = useState([]);
  const [cityTrends, setCityTrends] = useState([]);

  // Handle dropdown
  const handleChange = (e) => {
    const value = e.target.value;
    setSelectedOption(value);
    setSelectedData(value === "Province" ? data.province : data.areas);
  };

  // Fetch circle data
  useEffect(() => {
    fetch("/ml/city_circles")
      .then((res) => res.json())
      .then((data) => {
        console.log("✅ API returned:", data);        
        setCityCircles(data);        
      })
      .catch((err) => console.error("❌ Failed to fetch transactions", err));
  }, []);

  //Fetch and render the trend arrows
  useEffect(() => {
  fetch("http://localhost:8000/ml/city_price_trend")
    .then(res => res.json())
    .then(data => {
      console.log("📦 Trend data received:", data);
      if (Array.isArray(data)) {
        setCityTrends(data);
      } else {
        console.error("❌ Not an array:", data);
      }
    })
    .catch(err => console.error("❌ Fetch failed:", err));
}, []);


  // 🎯 Dynamic radius based on listings_count
  const getRadius = (count) => {
    if (count <= 100) return 14;
    else if (count > 100 && count <= 500) return 20;
    else if (count > 500 && count <= 1000) return 30;
    else if (count > 1000 && count <= 2000) return 40;
    else return 70;
  };

  return (
    <>
      <select
        name="Areas"
        id="lang"
        onChange={handleChange}
        value={selectedOption}
        style={{
          margin: 0,
          marginTop: 5,
          marginBottom: 5,
          padding: 8,
          border: 0,
          borderRadius: 10,
          fontWeight: 700,
        }}
      >
        <option value="Province">Province</option>
        <option value="District">Districts (Hottest)</option>
      </select>

      <MapContainer
        center={[33.993, 35.5]}
        zoom={9}
        style={{ height: "70vh" }}
        scrollWheelZoom={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* 🔵 Regular Data Markers */}
        {selectedData?.map((row, idx) => (
          <Marker key={idx} position={[row.latitude, row.longitude]}>
            <Popup>
              {selectedOption}: {row.province || row.city}
              <br />
              Listings: {row.listings_count}
            </Popup>
          </Marker>
        ))}

        {/* 🔴 Circles from city_prices (listings_count) */}
        {Array.isArray(cityCircles) && cityCircles.map((item, idx) => {
            const city = item.city?.trim() || "";
            const count = item.listings_count;
            const lat = item.latitude;
            const lng = item.longitude;

            if (!lat || !lng) {
              console.warn("❌ Missing coordinates for city:", city);
              return null;
            }

            return (
              <Circle
                key={`circle-${idx}`}
                center={[lat, lng]}
                radius={getRadius(count) * 100}
                pathOptions={{ color: "red", fillColor: "red", fillOpacity: 0.4 }}
              >
                <Tooltip>{city}: {count} listings</Tooltip>
              </Circle>
            );
          })}

        {/* 🔵 City Trend */}
        {Array.isArray(cityTrends) && cityTrends.map((item, idx) => {
          const { city, direction, latitude, longitude, change_percent } = item;

          const icon = L.icon({
            iconUrl:
              direction === "up"
                ? arrowUpTrend
                : direction === "down"
                ? arrowDownTrend
                : NeutralTrend,
            iconSize: [60, 60],
            iconAnchor: [20, 20],
          });

          if (!latitude || !longitude) return null;

          return (
            <Marker key={idx} position={[latitude, longitude]} icon={icon}>
              <Tooltip>
                {city}<br />
                Price Trend: {" "}
                {direction === "up"
                  ? "📈 Increasing"
                  : direction === "down"
                  ? "📉 Decreasing"
                  : "↔️ Stable"}
                <br />
                Change: {change_percent}%
              </Tooltip>
            </Marker>
          );
        })}

      </MapContainer>
    </>
  );
};

export default MapComponent;
