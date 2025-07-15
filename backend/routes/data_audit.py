import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_data_audit():
    """
    Performs a diagnostic audit on the properties.csv data to identify
    fundamental issues with consistency and quality.
    """
    # --- 0. Setup ---
    print("Starting Data Quality Audit...")
    # Create a directory to save our findings
    output_dir = "data_audit_report"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Report will be saved in '{output_dir}/'")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv('properties.csv')
    except FileNotFoundError:
        print("ERROR: properties.csv not found.")
        return

    # Basic cleaning for the audit
    df = df[(df['price_$'] > 100) & (df['size_m2'] > 10)].copy()
    for col in ['type', 'province', 'district']:
        df[col] = df[col].str.lower().str.strip()

    # --- 2. Hypothesis 1: Geospatial Sanity Check ---
    print("\n[Audit 1/3] Performing Geospatial Sanity Check...")
    
    # Select a few major districts to visualize
    districts_to_check = ['beirut', 'el metn', 'kesrouane', 'jbeil', 'batroun']
    df_geo_check = df[df['district'].isin(districts_to_check)]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_geo_check, x='longitude', y='latitude', hue='district', s=20, alpha=0.7)
    plt.title('Geospatial Distribution of Properties by District')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='District', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    geo_plot_path = os.path.join(output_dir, '1_geospatial_distribution.png')
    plt.savefig(geo_plot_path)
    plt.close()
    
    print(f"-> Geospatial plot saved to '{geo_plot_path}'")
    print("   Please inspect the plot. Are the dots for each district realistically spread out,")
    print("   or are they clumped at a single point? Clumping indicates bad coordinate data.")

    # --- 3. Hypothesis 2: Price vs. Size Correlation Check ---
    print("\n[Audit 2/3] Performing Price vs. Size Correlation Check...")
    
    # Focus on a common, well-represented group
    group_district = 'el metn'
    group_type = 'apartment'
    df_corr_check = df[(df['district'] == group_district) & (df['type'] == group_type)].copy()
    
    if not df_corr_check.empty:
        # We use log-log plot for better visualization of skewed data
        df_corr_check['log_price'] = np.log1p(df_corr_check['price_$'])
        df_corr_check['log_size_m2'] = np.log1p(df_corr_check['size_m2'])
        
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df_corr_check, x='log_size_m2', y='log_price',
                    scatter_kws={'alpha':0.4, 's':15}, line_kws={'color': 'red'})
        plt.title(f'Price vs. Size for {group_type.title()}s in {group_district.title()}')
        plt.xlabel('Log(Size in m²)')
        plt.ylabel('Log(Price in $)')
        corr_plot_path = os.path.join(output_dir, '2_price_vs_size_correlation.png')
        plt.savefig(corr_plot_path)
        plt.close()
        
        print(f"-> Price/Size correlation plot saved to '{corr_plot_path}'")
        print("   Please inspect the plot. Is there a general upward trend (as size increases, price increases)?")
        print("   A random cloud of points with no trend indicates severe data inconsistency.")
    else:
        print(f"-> Could not perform correlation check: No '{group_type}' in '{group_district}' found.")

    # --- 4. Hypothesis 3: Currency Anomaly Detection ---
    print("\n[Audit 3/3] Performing Currency Anomaly Detection...")
    
    # Calculate a rough 'price per square meter' to find suspicious values
    df['price_per_sqm'] = df['price_$'] / df['size_m2']
    
    # Find properties that are suspiciously cheap (e.g., < $100/sqm in prime areas)
    prime_districts = ['beirut', 'el metn']
    suspiciously_cheap = df[
        (df['district'].isin(prime_districts)) &
        (df['type'].isin(['apartment', 'house/villa'])) &
        (df['price_per_sqm'] < 100) # An apartment in Beirut for <$100/sqm is highly unlikely
    ].copy()

    # Save the findings to a text file
    report_path = os.path.join(output_dir, '3_suspiciously_cheap_listings.csv')
    if not suspiciously_cheap.empty:
        suspiciously_cheap.to_csv(report_path, index=False)
        print(f"-> Found {len(suspiciously_cheap)} potentially mis-priced (LBP?) listings.")
        print(f"   A report has been saved to '{report_path}'. Please review these listings.")
    else:
        print("-> No listings found that are suspiciously cheap in prime areas. This is a good sign.")

    print("\n--- AUDIT COMPLETE ---")
    print("Please review the generated files in the 'data_audit_report' directory.")

if __name__ == '__main__':
    run_data_audit()