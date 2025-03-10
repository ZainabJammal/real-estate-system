import React from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";

const MapComponent = ({ data }) => {
  return (
    <MapContainer center={[33.993, 35.5]} zoom={9} style={{ height: "70vh" }}>
      <TileLayer
        // attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {console.log(data)}
      {data?.map((row) => (
        <Marker position={[row?.latitude, row?.longitude]}>
          <Popup>
            District: {row?.city} <br /> Listings: {row?.listings_count}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
};

export default MapComponent;
