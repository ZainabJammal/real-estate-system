import pandas as pd
import re
import ast  


file_path = '/Users/amandamakdessi/Downloads/airbnb/airbnb_data_full.xlsx'
sheet_name = 'Result 1'
df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)


def extract_details(details):
    bedrooms = 'N/A'
    beds = 'N/A'
    bathrooms = 'N/A'

    if isinstance(details, str):  
        try:
            details = ast.literal_eval(details)  
        except (ValueError, SyntaxError):
            return bedrooms, beds, bathrooms  

    if isinstance(details, list):
        for item in details:
            item_lower = item.lower()
            if 'bedroom' in item_lower:
                bedrooms = re.search(r'\d+', item)
                if bedrooms:
                    bedrooms = bedrooms.group()
            elif 'bed' in item_lower and 'bedroom' not in item_lower:
                beds = re.search(r'\d+', item)
                if beds:
                    beds = beds.group()
            elif 'bath' in item_lower:
                bathrooms = re.search(r'\d+', item)
                if bathrooms:
                    bathrooms = bathrooms.group()
    
    return bedrooms, beds, bathrooms


data = []

for _, row in df.iterrows():
    address = row['Location'] if row['Location'] != 'Not Available' else 'N/A'
    rent_night = row['Price'] if pd.notna(row['Price']) else 'N/A'  
    
    
    if isinstance(rent_night, str):
        rent_night = rent_night.replace('$', '')
    
    bedrooms, beds, bathrooms = extract_details(row['Bed_Bath_Details'])
    
    guests = row['Guest'] if pd.notna(row['Guest']) else 'N/A'
    
    
    if isinstance(guests, str):
        guests = guests.replace(' guests', '')
    
    address = address.replace('Beriut', 'Beirut')
    
    data.append({
        'Address': address,
        'Rent $/night': rent_night,
        'Bedrooms': bedrooms,
        'Beds': beds,
        'Bathrooms': bathrooms,
        'Guests': guests
    })


new_df = pd.DataFrame(data)


new_df = new_df[(new_df['Address'] != 'N/A') & (new_df['Rent $/night'] != 'N/A')]


new_df.to_csv('/Users/amandamakdessi/Downloads/airbnb/cleanairbnbrent.csv', index=False)

print("CSV file has been created successfully.")