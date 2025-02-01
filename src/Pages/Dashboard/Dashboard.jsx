import React from "react";
<<<<<<< HEAD
import "./Dashboard.css";
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 0ec06ac (5 - Added New Pages (Explore Estates, Ask AI, Contact Agent) and added their corresponding styles)
import "./Page_Layout.css";
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
import Custom from "../../Components/CustomCard/Custom";
import Table from "../../Components/Table/Table";
import PieChart from "../../Components/PieChart/PieChartComponent";
import LineChartComponent from "../../Components/LineChart/LineChartComponent";
<<<<<<< HEAD
=======
=======
import "./Dashboard.css";
<<<<<<< HEAD
import Name from "../../Components/NameCard/Name";
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
=======
import Custom from "../../Components/CustomCard/Custom";
import Table from "../../Components/Table/Table";
import PieChart from "../../Components/PieChart/PieChartComponent";
<<<<<<< HEAD
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
=======
import LineChartComponent from "../../Components/LineChart/LineChartComponent";
>>>>>>> b815288 (3 - Removed Component CustomL and Created LineChartComponent instead and passed it as prop into Custom)
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30

function Dashboard() {
  return (
    <div className="dashboard-layout">
      <div className="dashboard-content">
        <div className="title">
          <h1>Dashboard</h1>
        </div>
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
        <div className="dashboard-components">
          <Custom
            title="Sales per Month"
            desc="This line chart shows the increasing amount of sales (in $) per each month"
            Component={LineChartComponent}
<<<<<<< HEAD
=======
=======
        <div className="dashboard-components">
          <Custom
            title="Sales per Month"
            desc="This line chart shows the increasing amount of sales (in $) per each month"
<<<<<<< HEAD
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
=======
            Component={LineChartComponent}
>>>>>>> b815288 (3 - Removed Component CustomL and Created LineChartComponent instead and passed it as prop into Custom)
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
          />

          <Custom
            title="Percentage of Highly Demanded Estates"
            desc="This chart represents the number of highly demanded estates according to locations"
            Component={PieChart}
          />
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
          <Custom
            title="Percentage of Highly Demanded Estates"
            desc="This chart represents the number of highly demanded estates according to locations"
            Component={PieChart}
          />
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
          <Custom />

>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
          <Custom />
          <Custom />
        </div>
        <div className="title">
          <h1>Tables</h1>
        </div>
        <div className="dashboard-components">
          <Custom
            type="table"
            title="Table of availabe estates"
            Component={Table}
          />
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
          <Custom />
          <Custom />
          <Custom
            title="Sales per Month"
            desc="This line chart shows the increasing amount of sales (in $) per each month"
            Component={LineChartComponent}
          />
          <Custom />
          <Custom />
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30

          <Custom
            type="table"
            title="Table of availabe estates"
            Component={Table}
<<<<<<< HEAD
            no_inflate
          />
        </div>
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 85ec564 (4 - Modified and stylized PieCharts and LineCharts to fit correctly on the dashboard (with different screens))
            no_inflate
          />
        </div>
=======
        <div className="dashboard-components"></div>
>>>>>>> df94a0d (1 - Created Dashboard layout and Components (Sidebar, NameCard, MenuCard))
=======
          />
        </div>
>>>>>>> 6bb4543 (2 - Added and Stylized New Components (Sidebar, Menu, Charts, etc..))
>>>>>>> 04ff9eb4cf35246afaacefd2a6e8b94cf9ac1c30
      </div>
    </div>
  );
}

export default Dashboard;
