import pandas as pd


file = "cleanairbnbrent.csv"
df = pd.read_csv(file)


valid_provinces = ["Mount Lebanon", "South", "Bekaa", "North", "Beirut", "Nabatieh", "Jabal Lubnan"]


def extract_province(address):
    for province in valid_provinces:
        if province in address:
            return "Mount Lebanon" if province == "Jabal Lubnan" else province
    return None


df["Province"] = df["Address"].apply(extract_province)


df_with_province = df.dropna(subset=["Province"])
df_without_province = df[df["Province"].isna()]


df_with_province.loc[:, "Rent $/night"] = pd.to_numeric(df_with_province["Rent $/night"], errors='coerce')
df_without_province.loc[:, "Rent $/night"] = pd.to_numeric(df_without_province["Rent $/night"], errors='coerce')


province_summary = df_with_province.groupby("Province")["Rent $/night"].agg(["mean", "median", "max", "min", "count"]).reset_index()
province_summary.columns = ["Province", "Avg Rent $/night", "Median Rent $/night", "Max Rent $/night", "Min Rent $/night", "Listings Count"]
province_summary[["Avg Rent $/night", "Median Rent $/night", "Max Rent $/night", "Min Rent $/night", "Listings Count"]] = province_summary[["Avg Rent $/night", "Median Rent $/night", "Max Rent $/night", "Min Rent $/night", "Listings Count"]].astype(int)


other_addresses_summary = df_without_province.groupby("Address")["Rent $/night"].agg(["mean", "median", "max", "min", "count"]).reset_index()
other_addresses_summary.columns = ["Address", "Avg Rent $/night", "Median Rent $/night", "Max Rent $/night", "Min Rent $/night", "Listings Count"]
other_addresses_summary[["Avg Rent $/night", "Median Rent $/night", "Max Rent $/night", "Min Rent $/night", "Listings Count"]] = other_addresses_summary[["Avg Rent $/night", "Median Rent $/night", "Max Rent $/night", "Min Rent $/night", "Listings Count"]].astype(int)


province_summary.to_csv("province_rentsummary.csv", index=False)
other_addresses_summary.to_csv("other_addresses_rentsummary.csv", index=False)

print("Sorting complete.")