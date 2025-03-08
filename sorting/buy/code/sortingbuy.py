import pandas as pd


file1 = "./scraping/realestate/clean_data/cleanrealestatebuy.csv"
file2 = "./scraping/jsk/clean_data/cleanjskbuy.csv"
file3 = "./sorting/dcoord.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)


df = pd.concat([df1, df2], ignore_index=True)


valid_provinces = ["Mount Lebanon", "South", "Bekaa", "North", "Beirut", "Nabatieh"]
df["Province"] = df["Address"].str.extract(f'({"|".join(valid_provinces)})')


valid_districts = [
    "Hasbaya", "Marjaayoun", "Nabatieh", "Bint Jbeil", "Beirut", "Minieh-danniyeh", "Bcharre", "Tripoli", "Koura", "Akkar", 
    "Zgharta", "Batroun", "Hermel", "West Bekaa", "Zahle", "Baalbeck", "Rachaiya", "Jezzine", "Tyre", "Saida", "Aley", 
    "Baabda", "Chouf", "El Metn", "Jbeil", "Kesrouane"]
df["District"] = df["Address"].str.split(",").str[1].str.strip()
df["District"] = df["District"].where(df["District"].isin(valid_districts))


df["City"] = df["Address"].str.split(",").str[0].str.strip()


df = df.dropna(subset=["Province"])


df["Price $"] = pd.to_numeric(df["Price $"], errors='coerce')


province_summary = df.groupby("Province")["Price $"].agg(["mean", "median", "max", "min", "count"]).reset_index()
province_summary.columns = ["Province", "Avg Price $", "Median Price $", "Max Price $", "Min Price $", "Listings Count"]
province_summary[["Avg Price $", "Median Price $", "Max Price $", "Min Price $", "Listings Count"]] = province_summary[["Avg Price $", "Median Price $", "Max Price $", "Min Price $", "Listings Count"]].astype(int)


district_summary = df.groupby("District")["Price $"].agg(["mean", "median", "max", "min", "count"]).reset_index()
district_summary = pd.merge(district_summary, df3, on="District", how="inner")
district_summary.columns = ["District", "Avg Price $", "Median Price $", "Max Price $", "Min Price $", "Listings Count", "Latitude", "Longitude"]
district_summary[["Avg Price $", "Median Price $", "Max Price $", "Min Price $", "Listings Count"]] = district_summary[["Avg Price $", "Median Price $", "Max Price $", "Min Price $", "Listings Count"]].astype(int)


city = df.groupby("City")["Price $"].agg(["mean", "median", "max", "min", "count"]).reset_index()
city.columns = ["City", "Avg Price $", "Median Price $", "Max Price $", "Min Price $", "Listings Count"]
city[["Avg Price $", "Median Price $", "Max Price $", "Min Price $", "Listings Count"]] = city[["Avg Price $", "Median Price $", "Max Price $", "Min Price $", "Listings Count"]].astype(int)


province_summary.to_csv("./sorting/buy/sorted_data/province_buysummary.csv", index=False)
district_summary.to_csv("./sorting/buy/sorted_data/district_buysummary.csv", index=False)
city.to_csv("./sorting/buy/sorted_data/city_buysummary.csv", index=False)

print("Sorting complete.")
