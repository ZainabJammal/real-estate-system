import requests
from bs4 import BeautifulSoup
import pandas as pd


base_url = "https://www.jskre.com/listings/residential-for-sale"


headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


total_pages = 301

properties = []


for page_number in range(1, total_pages + 1):
    url = f"{base_url}?page={page_number}"
    response = requests.get(url, headers=headers)

    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        main_section = soup.find('section', class_='max-w-[1366px] mx-auto')
        if main_section:
            for listing in main_section.find_all('div', class_='flex my-7 mx-2.5 flex-col min-[1090px]:flex-row'):
                address = listing.find('address', class_='poppins-light property-location flex items-center gap-[5px]')
                price = listing.find('p', class_='text-[#AD2F36] font-bold text-[18px] mt-2')
                property_type = listing.find('div', class_='max-[344px]:mr-[5px] max-[344px]:pr-[5px] pr-3 mr-3 fixInSmallScreen')

                size = "No Size"
                size_div = listing.find_all('div', class_="")
                for div in size_div:
                    size_text = div.get_text(strip=True)
                    if 'm²' in size_text:
                        size = size_text
                        break

                bedrooms_div = listing.find('div', class_='max-[344px]:mr-[5px] max-[344px]:pr-[5px] pr-2 mr-3')
                bedrooms = "No Bedrooms"
                if bedrooms_div:
                    bedrooms_text = bedrooms_div.text.strip()
                    if bedrooms_text:
                        bedrooms = bedrooms_text.split(" ")[0]

                bathrooms_div = listing.find_all('div', class_='max-[344px]:mr-[5px] max-[344px]:pr-[5px] pr-2 mr-3')
                bathrooms = "No Bathrooms"
                if len(bathrooms_div) > 1:
                    bathrooms_text = bathrooms_div[1].text.strip()
                    if bathrooms_text:
                        bathrooms = bathrooms_text.split(" ")[0]

                properties.append({
                    'Address': address.text.strip() if address else "No Address",
                    'Price': price.text.strip() if price else "No Price",
                    'Type': property_type.text.strip() if property_type else "No Type",
                    'Size': size,
                    'Bedrooms': bedrooms,
                    'Bathrooms': bathrooms
                })


if properties:
    df = pd.DataFrame(properties)
    df.to_csv('JSKRE.csv', index=False)
    print("Data extracted and saved to JSKRE.csv")
else:
    print("No listings found or unable to extract data.")