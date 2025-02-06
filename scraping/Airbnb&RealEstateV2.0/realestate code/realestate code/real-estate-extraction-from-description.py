import requests
import json
import csv
import os
import logging
import re  # For regular expressions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the URL and headers
url = "https://www.realestate.com.lb/laravel/api/member/properties"
headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Connection": "keep-alive",
    "Referer": "https://www.realestate.com.lb/en/rent-apartment-house-lebanon",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "X-XSRF-TOKEN": os.getenv("XSRF_TOKEN", "default_token")
}

cookies = {
    'XSRF-TOKEN': os.getenv("XSRF_TOKEN", "default_token"),
    'realestatecomlb_session': os.getenv("REALESTATE_SESSION", "default_session"),
}

# Define CSV file path and fieldnames
csv_file = "realestate_rent_5_data.csv"
fieldnames = [
    "title_en", "price", "location_en", "bedrooms", "bathrooms", "size_sqm", "description_en", "url"
]

def fetch_properties(page):
    params = {
        'pg': page,
        'sort': 'listing_level',
        'ct': '1',
        'direction': 'asc',
    }
    try:
        response = requests.get(url, headers=headers, params=params, cookies=cookies)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for page {page}: {e}")
        return None

def extract_values_from_description(description):
    """
    Extract bedrooms, bathrooms, and size_sqm from the description using regex.
    """
    # Regex patterns
    bedrooms_pattern = r"(\d+)\s*(?:bedrooms|BR|Beds|Bed)"
    bathrooms_pattern = r"(\d+)\s*(?:bathrooms|BA|Baths|Bath)"
    size_pattern = r"(\d+)\s*(?:sqm|m²|sq\.? m)"

    # Extract values
    bedrooms = re.search(bedrooms_pattern, description, re.IGNORECASE)
    bathrooms = re.search(bathrooms_pattern, description, re.IGNORECASE)
    size_sqm = re.search(size_pattern, description, re.IGNORECASE)

    # Return extracted values or "N/A" if not found
    return {
        "bedrooms": bedrooms.group(1) if bedrooms else "N/A",
        "bathrooms": bathrooms.group(1) if bathrooms else "N/A",
        "size_sqm": size_sqm.group(1) if size_sqm else "N/A",
    }

def extract_location_from_url_or_title(url, title):
    """
    Extract location from the URL or title.
    """
    # Try to extract location from URL
    location = re.search(r"/([^/]+)-lebanon$", url, re.IGNORECASE)
    if location:
        return location.group(1).replace("-", " ").title()  # Convert to title case

    # Try to extract location from title
    location = re.search(r"\b(?:in|at)\s+([\w\s]+)$", title, re.IGNORECASE)
    if location:
        return location.group(1).strip().title()  # Convert to title case

    return "N/A"  # Default if location not found

def extract_property_data(property):
    # Debug: Print the property JSON to inspect the structure
    # print(json.dumps(property, indent=4))

    # Extract values from description
    description = property.get("description_en", "")
    extracted_values = extract_values_from_description(description)

    # Extract location from URL or title
    location = extract_location_from_url_or_title(property.get("url", ""), property.get("title_en", ""))

    return {
        "title_en": property.get("title_en", "N/A"),
        "price": property.get("price", "N/A"),
        "location_en": location,  # Extracted from URL or title
        "bedrooms": extracted_values["bedrooms"],  # Extracted from description
        "bathrooms": extracted_values["bathrooms"],  # Extracted from description
        "size_sqm": extracted_values["size_sqm"],  # Extracted from description
        "description_en": description,
        "url": property.get("url", "N/A"),
    }

def main():
    all_properties = []
    page = 1
    total_pages = None  # Initialize total_pages as unknown

    while True:
        logging.info(f"Fetching data from page {page}...")
        result = fetch_properties(page)
        if not result or "data" not in result:
            logging.info("No more data found or invalid response.")
            break

        # Extract properties from the current page
        properties = result["data"].get("boostedProperties", []) + result["data"].get("docs", [])
        if not properties:
            logging.info("No properties found on this page.")
            break

        # Add properties to the list
        all_properties.extend([extract_property_data(prop) for prop in properties])

        # Check if total_pages is available in the response
        if total_pages is None and "total_pages" in result["data"]:
            total_pages = result["data"]["total_pages"]
            logging.info(f"Total pages to fetch: {total_pages}")

        # Break the loop if we've reached the last page
        if total_pages and page >= total_pages:
            logging.info("Reached the last page.")
            break

        page += 1  # Move to the next page

    if all_properties:
        with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_properties)
        logging.info(f"Data successfully written to {csv_file} (Total properties: {len(all_properties)})")
    else:
        logging.warning("No properties found to write to CSV.")

if __name__ == "__main__":
    main()