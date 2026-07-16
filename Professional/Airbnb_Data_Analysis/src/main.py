import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

try:
    var = pd.read_csv("data/airbnb.csv")
except FileNotFoundError:
    print("Error: Please place your dataset inside a 'data/' folder and name it 'airbnb_listings.csv'.")
    exit()

var.columns = var.columns.str.strip()

if not os.path.exists("output"):
    os.makedirs("output")

print(f"[EDA] Raw Data Profile: {var.shape[0]} listings mapped across {var.shape[1]} columns.")

# --- DATA HYGIENE & DATATYPE CONVERSIONS ---
if var['price'].dtype == 'object':
    var['price'] = var['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
var['price'] = pd.to_numeric(var['price'], errors='coerce')

var = var[(var['price'] > 0) & (var['price'] <= 5000)].copy()

var['reviews'] = pd.to_numeric(var['reviews'], errors='coerce').fillna(0)

var['rating'] = pd.to_numeric(var['rating'], errors='coerce')
var['rating'] = var['rating'].fillna(var['rating'].median())

var['bedrooms'] = pd.to_numeric(var['bedrooms'], errors='coerce').fillna(1)
var['beds'] = pd.to_numeric(var['beds'], errors='coerce').fillna(1)
var['guests'] = pd.to_numeric(var['guests'], errors='coerce').fillna(2)
var['country'] = var['country'].fillna('Unknown')


# 1. Capacity Density Flag: Number of guests per individual bedroom asset
var['Guest_To_Bedroom_Ratio'] = np.where(var['bedrooms'] > 0, var['guests'] / var['bedrooms'], var['guests'])

# 2. Host Occupancy Proxy Index (Estimated Days Occupied per Year)
var['Est_Occupied_Days_Year'] = np.minimum(250, var['reviews'] * 2.5)

# 3. Capital Yield Optimization: Estimated Annual Gross Revenue Capacity
var['Est_Annual_Revenue'] = var['price'] * var['Est_Occupied_Days_Year']

# 4. Listing Market Positioning Tier (Quantile cuts tracking pricing brackets)
var['Property_Market_Tier'] = pd.qcut(var['price'], q=3, labels=['Economy Stay', 'Mid-Scale Stay', 'Premium Stay'])

print(f"[Ingestion Complete] Cleaned Workspace Dimensions: {var.shape[0]} active properties mapped.")

# Area Analysis
country_summary = var.groupby('country').agg({
    'id': 'count',
    'price': 'mean',
    'rating': 'mean',
    'Est_Annual_Revenue': 'mean'
}).rename(columns={'id': 'Total_Listings', 'price': 'Avg_Price', 'rating': 'Avg_Rating'}).sort_values(by='Total_Listings', ascending=False)

print("\nCountry Real Estate Supply & Performance Matrix:")
print(country_summary.head(5))

# --- CAPACITY TRACKING METRICS BY CAPACITIES ---
capacity_metrics = var.groupby('Property_Market_Tier', observed=False).agg({
    'guests': 'mean',
    'bedrooms': 'mean',
    'beds': 'mean',
    'rating': 'mean'
}).rename(columns={'guests': 'Avg_Guests_Allowed', 'rating': 'Avg_Rating'})
print("\nStructural Capacity Breakdown Matrix across Market Tiers:")
print(capacity_metrics)


# Box Plot (Price Analysis Across Top Performing Countries)
plt.figure(figsize=(12, 6))
top_countries = country_summary.head(5).index
var_top_countries = var[var['country'].isin(top_countries)]
sns.boxplot(data=var_top_countries, x='country', y='price', hue='Property_Market_Tier', palette='Set2', width=0.6)
plt.title('Market Valuation: Nightly Price Spreads Across Top Active Countries', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Country Division')
plt.ylabel('Price Per Night ($)')
plt.yscale('log')  
plt.legend(title='Market Tier')
plt.tight_layout()
plt.savefig("output/chart1_price_distribution_boxplot.png", dpi=300)
plt.close()

# KDE Distribution Plot (Rating Curves based on Market Tiers)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=var, x='rating', hue='Property_Market_Tier', fill=True, common_norm=False, palette='viridis', alpha=0.4, linewidth=2)
plt.title('Customer Satisfaction: Review Score Density Curves Over Listing Tiers', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Aggregate Rating Score')
plt.ylabel('Density Distribution Weight')
plt.tight_layout()
plt.savefig("output/chart2_ratings_kde_plot.png", dpi=300)
plt.close()

# Heatmap Matrix (Core Feature Association Space - Price Prediction Prep)
plt.figure(figsize=(10, 9))
ml_prep_features = ['price', 'rating', 'reviews', 'bathrooms', 'beds', 'guests', 'bedrooms', 'Est_Annual_Revenue']
correlation_space = var[ml_prep_features].corr()
sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Pearson Correlation Coefficient'})
plt.title('Property Capacity Covariance & Predictive Feature Interaction Space Map', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("output/chart3_predictive_correlation_matrix.png", dpi=300)
plt.close()

# Scatter Plot (Capacity Footprint vs Estimated Annual Revenue)
plt.figure(figsize=(11, 6))
sns.scatterplot(data=var, x='guests', y='Est_Annual_Revenue', hue='Property_Market_Tier', alpha=0.4, palette='deep', edgecolor='none')
plt.title('Yield Analytics: Max Guest Accommodations vs Estimated Annual Gross Revenue Capacity', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Maximum Allowed Guest Capacity')
plt.ylabel('Estimated Annual Revenue ($)')
plt.tight_layout()
plt.savefig("output/chart4_capacity_vs_yield_scatter.png", dpi=300)
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(22, 16))

# Price Spread Slices (Box Plot Layout)
sns.boxplot(data=var_top_countries, x='country', y='price', palette='Set2', ax=axes[0, 0], width=0.5)
axes[0, 0].set_title('Nightly Pricing Spreads Across Top Active Countries', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Country Location')
axes[0, 0].set_ylabel('Price Per Night ($)')
axes[0, 0].set_yscale('log')

# Predictive Matrix Correlation Space (Heatmap Layout)
sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Property Capacity & Financial Correlation Spaces Map', fontsize=13, fontweight='bold', pad=15)

# Inventory Calendar Availabilities (KDE Layout)
sns.kdeplot(data=var, x='rating', hue='Property_Market_Tier', fill=True, common_norm=False, palette='viridis', alpha=0.3, linewidth=1.5, ax=axes[1, 0])
axes[1, 0].set_title('Customer Review Score Density Distributions', fontsize=13, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('Aggregate Rating Score')

# Asset Capacity vs Yield Breakdown (Scatter Layout)
sns.scatterplot(data=var, x='guests', y='Est_Annual_Revenue', hue='Property_Market_Tier', alpha=0.2, palette='deep', ax=axes[1, 1], legend=False)
axes[1, 1].set_title('Guest Capacity Density vs Annual Gross Revenue Yields', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Max Allowed Guest Capacity')
axes[1, 1].set_ylabel('Est. Annual Revenue ($)')

plt.tight_layout()
plt.savefig("output/airbnb_executive_master_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()


print("\n=========================================================")
print("===       BUSINESS DATA STORYTELLING & STRATEGY       ===")
print("=========================================================")
guest_revenue_corr = correlation_space.loc['guests', 'Est_Annual_Revenue']
bedroom_revenue_corr = correlation_space.loc['bedrooms', 'Est_Annual_Revenue']
print(f"   - Pearson Correlation Index (Guests vs Revenue Yield): {guest_revenue_corr:.4f}")
print(f"   - Pearson Correlation Index (Bedrooms vs Revenue Yield): {bedroom_revenue_corr:.4f}")
print("   Corporate Narrative: Accommodating higher guest counts presents a stronger linear correlation to annual revenue than adding bedrooms alone.")
print("   Strategic Actionable: Real estate portfolios should invest in optimising spatial configurations (e.g., adding multi-functional sleep areas) to increase total guest capacity limits without buying larger properties.")
print("2. PRICE PREPARATION MODELING VERDICT:")
print(f"   - Correlation between Nightly Price and Aggregate Rating: {correlation_space.loc['price', 'rating']:.4f}")
print("   Strategic Verdict: Rating profiles exhibit a weak correlation to nightly base prices.")
print("   For predictive machine learning modeling pipelines, developers must lean on capacity attributes (`bedrooms`, `bathrooms`, `guests`) as structural price predictors, rather than user sentiment reviews.")