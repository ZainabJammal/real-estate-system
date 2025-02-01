import requests
import pandas as pd
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BASE_URL = "https://www.realestate.com.lb/laravel/api/member/properties"
HEADERS = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Referer': 'https://www.realestate.com.lb/en/buy-apartment-house-lebanon?pg=1&sort=featured&ct=1',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'X-KL-kfa-Ajax-Request': 'Ajax_Request',
    'X-Requested-With': 'XMLHttpRequest',
    'X-XSRF-TOKEN': 'eyJpdiI6IjlHNWlDSCt0am5pclZvWnduUzJJUFE9PSIsInZhbHVlIjoiUnVqYXNKM01JcEQzMVdEcVBCdko4a242cmxFS1RGNTRhZjVWdXZaMzRhVEkzd08zMmcwNjVwbHVoT2Y2RHkwWGIwQzdTK2l6V0R6d0RpT2NLVkdNSmRnMkM3NTVpd2RuaUJlREVzOHphK2xScitJY0pkRitKZkxjU3kxa2pmbGMiLCJtYWMiOiJmMmI0MGRmNjM3NTY1N2JkMGUyN2IyNjY5NDRkNTAwNjAzNTBjM2RkNTM0YTBkMjIwYjhjOGVkOTdhZGE5YWM3IiwidGFnIjoiIn0=',
    'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}
COOKIES = {
    'XSRF-TOKEN': 'eyJpdiI6IjlHNWlDSCt0am5pclZvWnduUzJJUFE9PSIsInZhbHVlIjoiUnVqYXNKM01JcEQzMVdEcVBCdko4a242cmxFS1RGNTRhZjVWdXZaMzRhVEkzd08zMmcwNjVwbHVoT2Y2RHkwWGIwQzdTK2l6V0R6d0RpT2NLVkdNSmRnMkM3NTVpd2RuaUJlREVzOHphK2xScitJY0pkRitKZkxjU3kxa2pmbGMiLCJtYWMiOiJmMmI0MGRmNjM3NTY1N2JkMGUyN2IyNjY5NDRkNTAwNjAzNTBjM2RkNTM0YTBkMjIwYjhjOGVkOTdhZGE5YWM3IiwidGFnIjoiIn0%3D',
    'realestatecomlb_session': 'eyJpdiI6ImJVYUVzYkprOXNQRjF5WnVrWE91Q0E9PSIsInZhbHVlIjoiZUdQTkR4NmFxanppTGxtR25HTlVwQlVNS2pRRStZVU85UTFZUnYxV0p5U2Y0UXA1TC9VaFgxdjBpbEdoOWdoZEIwb2lJOVZ3NC80WjM3TWo4Um1wdFBLaFIvQWNiZlBHUlR5cXhsSmVydHNJUHRwOHc0Tnk5RVdqUG9TTXpwOXciLCJtYWMiOiJiYWVjZDk3ZTUyZDBlZDczMjk3ZWIzY2YyMjYyN2E0YjMxYTdmYTVlY2UyODYyNzZhYTE0M2UzNjhjZDQ4YWE3IiwidGFnIjoiIn0%3D',
}

# Initialize lists to store property data
properties_names = []
properties_price = []
properties_bathroom = []
properties_bedroom = []
properties_area = []
properties_districtID = []
properties_provinceID = []
properties_furnished = []


# Function to fetch data from a single page
def fetch_page(page):
    params = {
        'pg': str(page),
        'sort': 'listing_level',
        'ct': '1',
        'direction': 'asc',
    }
    try:
        response = requests.get(BASE_URL, headers=HEADERS, params=params, cookies=COOKIES)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching page {page}: {e}")
        return None


# Function to extract data from JSON response
def extract_data(result):
    if not result or "data" not in result:
        return

    properties = result["data"].get("boostedProperties", []) + result["data"].get("docs", [])

    for property in properties:
        properties_names.append(property.get("title_en", ""))
        properties_price.append(property.get("price", ""))
        properties_bathroom.append(property.get("bathroom_value", ""))
        properties_bedroom.append(property.get("bedroom_value", ""))
        properties_area.append(property.get("area", ""))
        properties_districtID.append(property.get("district_id", ""))
        properties_furnished.append(property.get("furnished", ""))
        properties_provinceID.append(property.get("province_id", ""))


# Main loop to fetch all pages
for page in range(1, 243):
    logging.info(f"Fetching page {page}")
    result = fetch_page(page)
    if result:
        extract_data(result)
    time.sleep(1)  # Add a delay to avoid overwhelming the server

# Create DataFrame and save to CSV
df = pd.DataFrame({
    "Title": properties_names,
    "Price": properties_price,
    "Bedroom": properties_bedroom,
    "Bathroom": properties_bathroom,
    "Area sqm": properties_area,
    "DistrictID": properties_districtID,
    "Furnished": properties_furnished,
    "ProvinceID": properties_provinceID
})

df.to_csv("realestate.csv", index=False)
logging.info("Data saved to realestate.csv")
