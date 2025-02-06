import pandas as pd


file = "cleanjskrent.csv"
df = pd.read_csv(file)


valid_provinces = ["Mount Lebanon", "South", "Bekaa", "North", "Beirut", "Nabatieh"]
valid_districts = [
    "Hasbaya", "Marjaayoun", "Nabatieh", "Bint Jbeil", "Beirut", "Minieh-danniyeh", "Bcharre", "Tripoli", "Koura", "Akkar", 
    "Zgharta", "Batroun", "Hermel", "West Bekaa", "Zahle", "Baalbeck", "Rachaiya", "Jezzine", "Tyre", "Saida", "Aley", 
    "Baabda", "Chouf", "El Metn", "Jbeil", "Kesrouane"
]


df["Province"] = df["Address"].str.extract(f'({"|".join(valid_provinces)})')
df["District"] = df["Address"].str.split(",").str[1].str.strip()
df["District"] = df["District"].where(df["District"].isin(valid_districts))
df["City"] = df["Address"].str.split(",").str[0].str.strip()


df = df.dropna(subset=["Province"])


df["Rent $/year"] = pd.to_numeric(df["Rent $/year"], errors='coerce')


province_summary = df.groupby("Province")["Rent $/year"].agg(["mean", "median", "max", "min", "count"]).reset_index()
province_summary.columns = ["Province", "Avg Rent $/year", "Median Rent $/year", "Max Rent $/year", "Min Rent $/year", "Listings Count"]
province_summary[["Avg Rent $/year", "Median Rent $/year", "Max Rent $/year", "Min Rent $/year", "Listings Count"]] = province_summary[["Avg Rent $/year", "Median Rent $/year", "Max Rent $/year", "Min Rent $/year", "Listings Count"]].astype(int)


district_summary = df.groupby("District")["Rent $/year"].agg(["mean", "median", "max", "min", "count"]).reset_index()
district_summary.columns = ["District", "Avg Rent $/year", "Median Rent $/year", "Max Rent $/year", "Min Rent $/year", "Listings Count"]
district_summary[["Avg Rent $/year", "Median Rent $/year", "Max Rent $/year", "Min Rent $/year", "Listings Count"]] = district_summary[["Avg Rent $/year", "Median Rent $/year", "Max Rent $/year", "Min Rent $/year", "Listings Count"]].astype(int)


city_summary = df.groupby("City")["Rent $/year"].agg(["mean", "median", "max", "min", "count"]).reset_index()
city_summary.columns = ["City", "Avg Rent $/year", "Median Rent $/year", "Max Rent $/year", "Min Rent $/year", "Listings Count"]
city_summary[["Avg Rent $/year", "Median Rent $/year", "Max Rent $/year", "Min Rent $/year", "Listings Count"]] = city_summary[["Avg Rent $/year", "Median Rent $/year", "Max Rent $/year", "Min Rent $/year", "Listings Count"]].astype(int)


province_summary.to_csv("province_rentsummary.csv", index=False)
district_summary.to_csv("district_rentsummary.csv", index=False)
city_summary.to_csv("city_rentsummary.csv", index=False)

print("Sorting complete.")