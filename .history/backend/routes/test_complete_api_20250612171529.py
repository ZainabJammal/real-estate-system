import requests
import json

# Test the historical data and LSTM forecasting endpoints
def test_historical_and_forecast():
    base_url = "http://localhost:5000"
    
    print("=== Testing Historical Data Endpoint ===")
    # Test historical data for single city
    historical_data = {
        "city": "Beirut",
        "granularity": "Y",
        "start_year": 2012,
        "end_year": 2016
    }
    
    try:
        response = requests.post(f"{base_url}/api/historical_data", json=historical_data)
        print(f"Historical Data Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Historical data points: {len(data.get('historical', []))}")
            if data.get('historical'):
                print(f"Sample historical data: {data['historical'][0]}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on localhost:5000")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Testing All Historical Data Endpoint ===")
    # Test historical data for all cities
    all_historical_data = {
        "granularity": "Y",
        "start_year": 2012,
        "end_year": 2016
    }
    
    try:
        response = requests.post(f"{base_url}/api/all_historical_data", json=all_historical_data)
        print(f"All Historical Data Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            cities = list(data.get('historical', {}).keys())
            print(f"Cities with historical data: {len(cities)}")
            if cities:
                print(f"Sample cities: {cities[:3]}")
                first_city = cities[0]
                print(f"Data points for {first_city}: {len(data['historical'][first_city])}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on localhost:5000")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Testing LSTM Forecast Endpoint ===")
    # Test LSTM forecasting
    lstm_data = {
        "granularity": "Y"
    }
    
    try:
        response = requests.post(f"{base_url}/api/lstm_forecast", json=lstm_data)
        print(f"LSTM Forecast Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Granularity: {data.get('granularity')}")
            print(f"Forecast Periods: {data.get('forecast_periods')}")
            print(f"Number of cities: {len(data.get('forecasts', {}))}")
            
            # Print first city's forecast as example
            forecasts = data.get('forecasts', {})
            if forecasts:
                first_city = list(forecasts.keys())[0]
                print(f"\nSample forecast for {first_city}:")
                for i, forecast in enumerate(forecasts[first_city][:3]):  # Show first 3 predictions
                    print(f"  {forecast['date']}: {forecast['predicted_value']}")
                if len(forecasts[first_city]) > 3:
                    print(f"  ... and {len(forecasts[first_city]) - 3} more predictions")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on localhost:5000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_historical_and_forecast()

