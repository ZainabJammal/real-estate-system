import { useEffect, useState } from "react";
import HeatMap from "react-heatmap-grid";

const Heatmap = ({ data }) => {
  const [xLabels, setxLabels] = useState([]);
  const [yLabels, setyLabels] = useState([]);
  useEffect(() => {
    if (data) {
      setxLabels(data?.map((row) => row.city));
      setyLabels(data?.map((row) => row.type));
    }
  }, []);


  return (
    <>
      <HeatMap
        xLabels={xLabels}
        yLabels={yLabels}
        data={data}
        background="white"
        cellStyle={(value) => ({
          background: `rgb(255, ${255 - value * 5}, ${255 - value * 5})`,
          fontSize: "14px",
        })}
      />
    </>
  );
};

export default Heatmap;
// Note: The Heatmap component is designed to visualize data in a grid format.
// It uses the `react-heatmap-grid` library to create a heatmap based on the provided data.
