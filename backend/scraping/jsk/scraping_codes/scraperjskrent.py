import requests
from bs4 import BeautifulSoup
import pandas as pd


base_url = "https://www.jskre.com/listings/for-rent"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

total_pages = 400

properties = []

for page_number in range(1, total_pages + 1):
    url = f"{base_url}?page={page_number}"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        main_section = soup.find('section', class_='max-w-[1366px] mx-auto')
        if main_section:
            listings = main_section.find_all('div', class_='flex my-7 mx-2.5 flex-col min-[1090px]:flex-row')
            
            for listing in listings:
                
                address = listing.find('address', class_='poppins-light property-location flex items-center gap-[5px]')
                address_text = address.text.strip() if address else "N/A"
                
                price = listing.find('p', class_='text-[#AD2F36] font-bold text-[18px] mt-2')
                price_text = price.text.strip() if price else "N/A"
                
                property_type = listing.find('div', class_='max-[344px]:mr-[5px] max-[344px]:pr-[5px] pr-3 mr-3 fixInSmallScreen')
                property_type = property_type.text.strip() if property_type else "N/A"

                
                size = "N/A"
                size_div = listing.find_all('div', class_="")
                for div in size_div:
                    size_text = div.get_text(strip=True)
                    if 'm²' in size_text:
                        size = size_text
                        break
                        

                
                bedrooms = "N/A"
                bedrooms_div = listing.find('div', class_='max-[344px]:mr-[5px] max-[344px]:pr-[5px] pr-2 mr-3')
                if bedrooms_div:
                    bedrooms_text = bedrooms_div.text.strip()
                    if bedrooms_text:
                        bedrooms = bedrooms_text.split(" ")[0]

                
                bathrooms = "N/A"
                bathrooms_div = listing.find_all('div', class_='max-[344px]:mr-[5px] max-[344px]:pr-[5px] pr-2 mr-3')
                if len(bathrooms_div) > 1:
                    bathrooms_text = bathrooms_div[1].text.strip()
                    if bathrooms_text:
                        bathrooms = bathrooms_text.split(" ")[0]
                
                
                properties.append({
                    
                    "Address": address_text,
                    "Price": price_text,
                    "Type": property_type,
                    "Size": size,
                    "Bedrooms": bedrooms,
                    "Bathrooms": bathrooms,
                    
                    
                })
                print(f"Data extracted from page {page_number}")


rental_df = pd.DataFrame(properties)
rental_df.to_csv("./scraping/jsk/full_data/JSKREfullrent.csv", index=False)

print("Scraping completed. Data saved to JSKREfullrent.csv")
