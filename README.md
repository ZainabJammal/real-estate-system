# Real Estate Dashboard System

## Description

This project is an end-to-end dashboard system built with **React** (via **Vite**) for the frontend and **Quart** for the backend. It scrapes data from various websites, cleans and processes it, and stores it in a **Supabase** database. The frontend fetches the data via API calls and displays it using **Recharts.js**.

## Table of Contents

1. [Installation](#installation)
2. [Running the Backend](#running-the-backend)
3. [Running the Frontend](#running-the-frontend)
4. [Folder Structure](#folder-structure)
5. [Technologies Used](#technologies-used)
6. [License](#license)

## Installation

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/real-estate-dashboard.git
cd real-estate-dashboard
```

## Set Up the Backend

### **A. Install Dependencies**

1. Navigate to the `backend` directory:

   ```bash
   cd backend
   ```

2. Create and activate the virtual environment:

   ```bash
   python -m venv .venv  # Create a virtual environment
   .\.venv\Scripts\activate  # Activate the virtual environment (on Windows)
   ```
   
3. Install the required Python dependencies by using the requirements.txt:

   ```bash
   pip install -r requirements.txt
   ```

### **B. Configure the Environment Variables**

   In the backend folder, create a .env file and add your Supabase credentials. Replace the placeholders with your actual credentials.

   ```bash
   SUPABASE_URL=your-supabase-url
   SUPABASE_KEY=your-supabase-key
   ```

### **C. Running the Backend Server**

   After activating the virtual environment and installing dependencies, start the backend server using Hypercorn (the ASGI server):

   ```bash
   python -m hypercorn server:app
   ```

## Set Up the Frontend

### 1. Install Dependencies

1. Navigate to the `src` directory (frontend):

   ```bash
   cd frontend/src
   ```
2. Install the required dependencies using npm or yarn:

   ```bash
   npm install
   ```
   This will install all necessary dependencies as specified in the package.json file for the frontend.

### 2. Running the Frontend

   Once the dependencies are installed, start the frontend development server using npm:

   ```bash
   npm run dev
   ```
The frontend will now be available at http://localhost:3000 (or the port specified in your .env file). Open this URL in your browser to see the real estate dashboard in action.
