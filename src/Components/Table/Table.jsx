import React from "react";
import "./Table.css";

function Table() {
  const dummy = [
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
    {
      FirstName: "Mostafa",
      LastName: "Dawi",
      Phone: 12345678,
      Address: "Address_1",
      Price: 50000,
    },
  ];
  return (
    <div className="table-card">
      <table>
        <thead>
          <tr>
            <th>FirstName</th>
            <th>LastName</th>
            <th>Phone</th>
            <th>Address</th>
            <th>Price</th>
          </tr>
        </thead>

        <tbody>
          {dummy?.map((row, _idx) => {
            return (
              <tr key={_idx}>
                <td>{row.FirstName}</td>
                <td>{row.LastName}</td>
                <td>{row.Phone}</td>
                <td>{row.Address}</td>
                <td>{row.Price}$</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default Table;
