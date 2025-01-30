import requests
import json
import pandas as pd

# ========================================================== REALESTATE.COM.LB ====================================================================
# cookies = {
#     'XSRF-TOKEN': 'eyJpdiI6IjlHNWlDSCt0am5pclZvWnduUzJJUFE9PSIsInZhbHVlIjoiUnVqYXNKM01JcEQzMVdEcVBCdko4a242cmxFS1RGNTRhZjVWdXZaMzRhVEkzd08zMmcwNjVwbHVoT2Y2RHkwWGIwQzdTK2l6V0R6d0RpT2NLVkdNSmRnMkM3NTVpd2RuaUJlREVzOHphK2xScitJY0pkRitKZkxjU3kxa2pmbGMiLCJtYWMiOiJmMmI0MGRmNjM3NTY1N2JkMGUyN2IyNjY5NDRkNTAwNjAzNTBjM2RkNTM0YTBkMjIwYjhjOGVkOTdhZGE5YWM3IiwidGFnIjoiIn0%3D',
#     'realestatecomlb_session': 'eyJpdiI6ImJVYUVzYkprOXNQRjF5WnVrWE91Q0E9PSIsInZhbHVlIjoiZUdQTkR4NmFxanppTGxtR25HTlVwQlVNS2pRRStZVU85UTFZUnYxV0p5U2Y0UXA1TC9VaFgxdjBpbEdoOWdoZEIwb2lJOVZ3NC80WjM3TWo4Um1wdFBLaFIvQWNiZlBHUlR5cXhsSmVydHNJUHRwOHc0Tnk5RVdqUG9TTXpwOXciLCJtYWMiOiJiYWVjZDk3ZTUyZDBlZDczMjk3ZWIzY2YyMjYyN2E0YjMxYTdmYTVlY2UyODYyNzZhYTE0M2UzNjhjZDQ4YWE3IiwidGFnIjoiIn0%3D',
# }

# headers = {
#     'Accept': 'application/json, text/plain, */*',
#     'Accept-Language': 'en-US,en;q=0.9',
#     'Connection': 'keep-alive',
#     'Referer': 'https://www.realestate.com.lb/en/buy-apartment-house-lebanon?pg=1&sort=featured&ct=1',
#     'Sec-Fetch-Dest': 'empty',
#     'Sec-Fetch-Mode': 'cors',
#     'Sec-Fetch-Site': 'same-origin',
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
#     'X-KL-kfa-Ajax-Request': 'Ajax_Request',
#     'X-Requested-With': 'XMLHttpRequest',
#     'X-XSRF-TOKEN': 'eyJpdiI6IjlHNWlDSCt0am5pclZvWnduUzJJUFE9PSIsInZhbHVlIjoiUnVqYXNKM01JcEQzMVdEcVBCdko4a242cmxFS1RGNTRhZjVWdXZaMzRhVEkzd08zMmcwNjVwbHVoT2Y2RHkwWGIwQzdTK2l6V0R6d0RpT2NLVkdNSmRnMkM3NTVpd2RuaUJlREVzOHphK2xScitJY0pkRitKZkxjU3kxa2pmbGMiLCJtYWMiOiJmMmI0MGRmNjM3NTY1N2JkMGUyN2IyNjY5NDRkNTAwNjAzNTBjM2RkNTM0YTBkMjIwYjhjOGVkOTdhZGE5YWM3IiwidGFnIjoiIn0=',
#     'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
#     'sec-ch-ua-mobile': '?0',
#     'sec-ch-ua-platform': '"Windows"',
# }

# params = (
#     ('pg', '1'),
#     ('sort', 'listing_level'),
#     ('ct', '1'),
#     ('direction', 'asc'),
# )

# url = "https://www.realestate.com.lb/laravel/api/member/properties"

# # Make the POST request
# response = requests.get(url, headers=headers, params=params, cookies=cookies)

# result = response.json()

# dict1 = {}
# i = 0
# for i, property in enumerate(result["data"]["boostedProperties"]):
#     dict1[i] = property
#     print(i)
# while (i < len(result["data"]["docs"])):
#     dict1[i] = result["data"]["docs"][i]
#     print(i)
#     i += 1


# # # Handle response
# # if response.status_code == 200:
# #     print("Success!\n")
# #     print(json.dumps(dict1, indent=4))
# #     # print(json.dumps(result.keys(), indent=4))  # Parse and print the JSON response
# # else:
# #     print(f"Failed with status code {response.status_code}")
# #     print(response.text)

# # print('{"index":"olx-lb-production-ads-en"}\n{"from":0,"size":0,"track_total_hits":false,"query"\n}')
# i = 0
# for property in dict1.values():
#     print("Title: "+property["title_en"]+"\t Price: $"+ str(property["price"]))


# =============================================================== DUBIZZLE ======================================================================= 

headers = {
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9',
    'authorization': 'Basic b2x4LWxiLXByb2R1Y3Rpb24tc2VhcmNoOj5zK08zPXM5QEk0REYwSWEldWc/N1FQdXkye0RqW0Zy',
    'content-type': 'application/x-ndjson',
    'origin': 'https://www.dubizzle.com.lb',
    'priority': 'u=1, i',
    'referer': 'https://www.dubizzle.com.lb/',
    'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'cross-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
}

data = '{"index":"olx-lb-production-ads-en"}\n{"from":0,"size":0,"track_total_hits":false,"query":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}}]}},"aggs":{"category.lvl1.externalID":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.lvl0.externalID":"138"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"category.lvl1.externalID","size":20}}}}}},"category.lvl2.externalID":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.lvl1.externalID":"95"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"category.lvl2.externalID","size":20}}}}}},"location.lvl1":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.lvl0.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"location.lvl1.externalID","size":20},"aggs":{"complex_value":{"top_hits":{"size":1,"_source":{"include":["location.lvl1"]}}}}}}}}},"extraFields.property_type":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.property_type","size":20}}}}}},"extraFields.video":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.video","size":20}}}}}},"extraFields.ownership":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.ownership","size":20}}}}}},"extraFields.panorama":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.panorama","size":20}}}}}},"extraFields.payment_option":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.payment_option","size":20}}}}}},"extraFields.rooms":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.rooms","size":20}}}}}},"extraFields.bathrooms":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.bathrooms","size":20}}}}}},"extraFields.furnished":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.furnished","size":20}}}}}},"extraFields.delivery_date":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.delivery_date","size":20}}}}}},"extraFields.floor_level":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.floor_level","size":20}}}}}},"extraFields.features":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.features","size":20}}}}}},"extraFields.new":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.new","size":20}}}}}},"extraFields.hot":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.hot","size":20}}}}}},"extraFields.verified":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.verified","size":20}}}}}},"extraFields.zero_km":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.zero_km","size":20}}}}}},"extraFields.discounted":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.discounted","size":20}}}}}},"extraFields.save_ten_percent":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.save_ten_percent","size":20}}}}}},"extraFields.save_twenty_percent":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.save_twenty_percent","size":20}}}}}},"extraFields.save_thirty_percent":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.save_thirty_percent","size":20}}}}}},"extraFields.save_forty_percent":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.save_forty_percent","size":20}}}}}},"extraFields.save_fifty_percent":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.save_fifty_percent","size":20}}}}}},"extraFields.save_sixty_percent":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.save_sixty_percent","size":20}}}}}},"extraFields.save_seventy_percent":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.save_seventy_percent","size":20}}}}}},"extraFields.eid_collection":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.eid_collection","size":20}}}}}},"extraFields.new_collection":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.new_collection","size":20}}}}}},"extraFields.black_friday":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.black_friday","size":20}}}}}},"extraFields.holidays":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.holidays","size":20}}}}}},"extraFields.ramadan":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.ramadan","size":20}}}}}},"extraFields.weekly_finds":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.weekly_finds","size":20}}}}}},"extraFields.highlights":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.highlights","size":20}}}}}},"extraFields.summer":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.summer","size":20}}}}}},"extraFields.autumn":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"extraFields.autumn","size":20}}}}}},"product":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}},{"bool":{"should":[{"term":{"product":"featured"}},{"term":{"product":"elite"}}]}}]}},"aggs":{"facet":{"terms":{"field":"product","size":20},"aggs":{"complex_value":{"top_hits":{"size":1,"_source":{"include":["product"]}}}}}}}}},"totalProductCount":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"product":"elite"}},{"term":{"product":"featured"}}]}},"aggs":{"facet":{"terms":{"field":"product","size":20},"aggs":{"complex_value":{"top_hits":{"size":1,"_source":{"include":["totalProductCount"]}}}}}}}}},"type":{"global":{},"aggs":{"filtered_agg":{"filter":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"location.externalID":"0-1"}}]}},"aggs":{"facet":{"terms":{"field":"type","size":20}}}}}}}}\n{"index":"olx-lb-production-ads-en"}\n{"from":0,"size":3,"track_total_hits":200000,"query":{"function_score":{"random_score":{"seed":664},"query":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"product":"elite"}}]}}}},"sort":["_score"]}\n{"index":"olx-lb-production-ads-en"}\n{"from":0,"size":8,"track_total_hits":200000,"query":{"function_score":{"random_score":{"seed":664},"query":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}},{"term":{"product":"featured"}}]}}}},"sort":["_score"]}\n{"index":"olx-lb-production-ads-en"}\n{"from":0,"size":34,"track_total_hits":200000,"query":{"bool":{"must":[{"term":{"category.slug":"apartments-villas-for-sale"}}],"must_not":[{"term":{"product":"elite"}}]}},"sort":[{"timestamp":{"order":"desc"}},{"id":{"order":"desc"}}]}\n'

# Wait for response from request
response = requests.post(
    'https://search.mena.sector.run/_msearch?filter_path=took%2C*.took%2C*.suggest.*.options.text%2C*.suggest.*.options._source.*%2C*.hits.total.*%2C*.hits.hits._source.*%2C*.hits.hits._score%2C*.hits.hits.highlight.*%2C*.error%2C*.aggregations.*.buckets.key%2C*.aggregations.*.buckets.doc_count%2C*.aggregations.*.buckets.complex_value.hits.hits._source%2C*.aggregations.*.filtered_agg.facet.buckets.key%2C*.aggregations.*.filtered_agg.facet.buckets.doc_count%2C*.aggregations.*.filtered_agg.facet.buckets.complex_value.hits.hits._source',
    headers=headers,
    data=data,
)

# Result stored as JSON
result = response.json()

# Test: Printing title and prices of properties scraped from dubizzle backend API
i = 1
count = 0

properties_names = []
properties_price = []

while(i < len(result["responses"])):
    properties = result["responses"][i]["hits"]["hits"]
    j = 0
    while j < len(properties):
        # print("{\n\tTitle: "+ properties[j]["_source"]["title"] + "\n\t Price: " + str(properties[j]["_source"]["formattedExtraFields"][1]["formattedValue"]) + "\n}")

        properties_names.append(properties[j]["_source"]["title"])
        properties_price.append(properties[j]["_source"]["formattedExtraFields"][1]["formattedValue"])

        j += 1
        count += 1

        
    i += 1

print(count)

df = pd.DataFrame({"Title": properties_names, "Price": properties_price})
df.to_csv("./extracted_data/dubizzle.csv")

