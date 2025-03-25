import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";

const MapComponent = ({ data }) => {
  const initialData = data.province;
  const [selectedData, setSelectedData] = useState(initialData);
  const [selectedOption, setSelectedOption] = useState("Province");
  useEffect(() => {
    if (selectedOption === "Province") {
      setSelectedData(data.province);
    } else {
      setSelectedData(data.areas);
    }
  }, [data, selectedOption]);

  const handleChange = (e) => {
    setSelectedOption(e.target.value);
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
        {console.log(data)}
        {selectedData?.map((row, _idx) => (
          <Marker key={_idx} position={[row?.latitude, row?.longitude]}>
            <Popup>
              {selectedOption}: {row?.province ? row?.province : row?.city}
              <br /> Listings: {row?.listings_count}
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </>
  );
};

export default MapComponent;
