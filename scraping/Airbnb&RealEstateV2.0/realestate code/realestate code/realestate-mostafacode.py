import requests
import json
import pandas as pd
# ========================================================== REALESTATE.COM.LB ====================================================================
cookies = {
    'XSRF-TOKEN': 'eyJpdiI6IjlHNWlDSCt0am5pclZvWnduUzJJUFE9PSIsInZhbHVlIjoiUnVqYXNKM01JcEQzMVdEcVBCdko4a242cmxFS1RGNTRhZjVWdXZaMzRhVEkzd08zMmcwNjVwbHVoT2Y2RHkwWGIwQzdTK2l6V0R6d0RpT2NLVkdNSmRnMkM3NTVpd2RuaUJlREVzOHphK2xScitJY0pkRitKZkxjU3kxa2pmbGMiLCJtYWMiOiJmMmI0MGRmNjM3NTY1N2JkMGUyN2IyNjY5NDRkNTAwNjAzNTBjM2RkNTM0YTBkMjIwYjhjOGVkOTdhZGE5YWM3IiwidGFnIjoiIn0%3D',
     'realestatecomlb_session': 'eyJpdiI6ImJVYUVzYkprOXNQRjF5WnVrWE91Q0E9PSIsInZhbHVlIjoiZUdQTkR4NmFxanppTGxtR25HTlVwQlVNS2pRRStZVU85UTFZUnYxV0p5U2Y0UXA1TC9VaFgxdjBpbEdoOWdoZEIwb2lJOVZ3NC80WjM3TWo4Um1wdFBLaFIvQWNiZlBHUlR5cXhsSmVydHNJUHRwOHc0Tnk5RVdqUG9TTXpwOXciLCJtYWMiOiJiYWVjZDk3ZTUyZDBlZDczMjk3ZWIzY2YyMjYyN2E0YjMxYTdmYTVlY2UyODYyNzZhYTE0M2UzNjhjZDQ4YWE3IiwidGFnIjoiIn0%3D',
 }

headers = {
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

params = (
     ('pg', '1'),
     ('sort', 'listing_level'),
     ('ct', '1'),
     ('direction', 'asc'),
 )

url = "https://www.realestate.com.lb/laravel/api/member/properties"

# # Make the POST request
response = requests.get(url, headers=headers, params=params, cookies=cookies)

result = response.json()

dict1 = {}
i = 0
for i, property in enumerate(result["data"]["boostedProperties"]):
     dict1[i] = property
     print(i)
while (i < len(result["data"]["docs"])):
     dict1[i] = result["data"]["docs"][i]
     print(i)
     i += 1


# # # Handle response
if response.status_code == 200:
     print("Success!\n")
     print(json.dumps(dict1, indent=4))
     # print(json.dumps(result.keys(), indent=4))  # Parse and print the JSON response
else:
     print(f"Failed with status code {response.status_code}")
     print(response.text)

     #print('{"index":"olx-lb-production-ads-en"}\n{"from":0,"size":0,"track_total_hits":false,"query"\n}')
i = 0
for property in dict1.values():
     print("Title: "+property["title_en"]+"\t Price: $"+ str(property["price"]))
     df = pd.DataFrame({"Title": property, "Price": str(property["price"])})
     df.to_csv("C:/Users/RAHAL/Downloads/airbnbScraping-main/realestateproperties.csv")