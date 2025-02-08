import React from "react";
import "./Table.css";

function Table({ data }) {
  if (!data || data.length === 0) return <p>No data available</p>;

  // Extract column names dynamically
  const columns = Object.keys(data[0]);
  return (
    <div className="table-card">
      <table border="1">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th> // Render column names dynamically
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {columns.map((col) => (
                <td key={col}>{row[col]}</td> // Render cell values dynamically
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default Table;
