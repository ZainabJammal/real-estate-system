import requests
import json
import csv
import time

# Base URL for the API endpoint
base_url = "https://www.realestate.com.lb/_next/data/tJD_gYeW-5efE5YO9Cx7J/en/real-estate-brokers-agents-lebanon/{page}.json?id={page}&_={timestamp}"

# Headers to mimic the curl request
headers = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Connection": "keep-alive",
    "Cookie": "_ga=GA1.1.693996357.1736874612; _gcl_au=1.1.1554251831.1736874612; _ga_HRCEDE2K8L=GS1.1.1738247956.12.1.1738249323.58.0.247075379; XSRF-TOKEN=eyJpdiI6IldBcHN4KzlCY2NrVjN3OVhadVJJdmc9PSIsInZhbHVlIjoiS2kvdFJ4YW5sazdUSmE4enM0M2ZhQ1VlbUNaSjUwa1FKTjl6YjBGRHo2TC9SeEN3T21tYkphaEZBMWE0U3h4d1M4TWg3NXhvUCtvK1JRQXIxdkhIc2xBUWluS1EyVkozUUFFYWNLWGFzakx6Rmw3Q2c1SndjNGVKTXFSVkg5WVUiLCJtYWMiOiJkMjZmOTgwMjYxNzM3ZTY0YzRkMDk4OWViNmVkMzJlMGYyNDM0OGY3MDk1NzFlYWQ5ODFkZjkyOTNjZDUxMTFjIiwidGFnIjoiIn0%3D; realestatecomlb_session=eyJpdiI6Im1Za00wZkxDVzR1SUs0bktqQ09vbUE9PSIsInZhbHVlIjoidlpSZE54a1VhR21uN21qdm5PSS92QmFQTGcrNHo1V3BFd3JJRTQwZmdKMWJvT20raFB4RmExVHB5SjlRL3NJOHhzdlRyVGxFeWptRzQ4aUdGalVBU0dVSC9oV01lU0sySUozbkw4L0txRGx5N3pER28wV2RrMWRMOU1UNWZKMkMiLCJtYWMiOiI1YTQwMjA3ZTkzYjhkNGNjYjg1MzVmNzc5MWI2OWIzMzBjYWNhODdlYjgxN2FjNGJiNzkxYjllYzU4ODUyYTkyIiwidGFnIjoiIn0%3D",
    "Referer": "https://www.realestate.com.lb/en/buy-apartment-house-lebanon/el-metn-district/apartment/CP-AK05",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "purpose": "prefetch",
    "sec-ch-ua": '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "x-nextjs-data": "1",
}

# Define CSV file path
csv_file = "locations_data2.csv"

# Define CSV column headers
fieldnames = [
    "id", "province_id", "name_en", "name_ar", "copy_field_en", "copy_field_ar"
]

# Open the CSV file for writing
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header row

    # Loop through all pages (1 to 243)
    for page in range(1, 243):
        print(f"Scraping page {page}...")

        # Construct the URL for the current page
        url = base_url.format(page=page, timestamp=int(time.time()))

        try:
            # Make the GET request
            response = requests.get(url, headers=headers)
            print(f"Status Code: {response.status_code}")  # Debugging: Check the status code

            response.raise_for_status()  # Raise an exception for HTTP errors

            # Try to parse the response as JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                print(f"Server returned non-JSON content for page {page}:")
                print(response.text)
                continue

            # Extract the locations data
            locations = data.get("pageProps", {}).get("locations", [])
            if not locations:
                print(f"No locations data found in the response for page {page}.")
                continue

            # Write each location to the CSV file
            for location in locations:
                row = {
                    "id": location.get("id", ""),
                    "province_id": location.get("province_id", ""),
                    "name_en": location.get("name_en", ""),
                    "name_ar": location.get("name_ar", ""),
                    "copy_field_en": location.get("copy_field_en", ""),
                    "copy_field_ar": location.get("copy_field_ar", ""),
                }
                writer.writerow(row)

            # Add a delay to avoid overwhelming the server
            time.sleep(2)  # Wait 2 seconds between requests

        except requests.exceptions.RequestException as e:
            print(f"Request failed for page {page}: {e}")
        except Exception as e:
            print(f"An error occurred for page {page}: {e}")

print(f"Scraping completed. Locations data successfully written to {csv_file}.")