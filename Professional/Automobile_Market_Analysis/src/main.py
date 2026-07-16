import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

print("=========================================================")
print("=== PHASE 1: ENTERPRISE DATA INGESTION & DATA CLEANING ===")
print("=========================================================")

try:
    df = pd.read_csv("data/automobile_data.csv")
except FileNotFoundError:
    print("Error: Please place your dataset inside a 'data/' folder and name it 'automobile_data.csv'.")
    exit()

df.columns = df.columns.str.strip()

os.makedirs("output", exist_ok=True)

print(f"[EDA] Raw Data Profile: {df.shape[0]} vehicles mapped across {df.shape[1]} columns.")

# --- STEP 1: CLEANING MISSING OR MASKED VALUES ---
# Real-world automotive datasets often use '?' to indicate missing data. We clean those up here:
for col in ['price', 'horsepower', 'num_of_doors']:
    if df[col].dtype == 'object':
        df[col] = df[col].replace('?', np.nan)

# Convert primary continuous fields to correct numeric datatypes
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['curb_weight'] = pd.to_numeric(df['curb_weight'], errors='coerce')

# Drop records lacking critical target baseline pricing data
df = df.dropna(subset=['price']).copy()

# Impute minor categorical gaps or mechanical markers safely using the column medians
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

# --- STEP 2: ADVANCED TALENT FEATURE ENGINEERING ---
# 1. Structural Brand Classification Tier: Stratify ultra-premium marques vs consumer mass market units
luxury_brands = ['bmw', 'mercedes-benz', 'audi', 'porsche', 'jaguar', 'volvo']
df['Brand_Tier'] = df['make'].str.lower().apply(lambda x: 'Luxury' if x in luxury_brands else 'Standard')

# 2. Weight-to-Power Capacity Index: Identifies performance vehicle dynamics
df['Weight_Per_HP'] = df['curb_weight'] / df['horsepower']

# 3. Market Pricing Bracket: Quantile cuts tracking asset market positioning strategy
df['Market_Bracket'] = pd.qcut(df['price'], q=3, labels=['Budget Asset', 'Mid-Range Asset', 'Premium Asset'])

print(f"[Ingestion Complete] Cleaned Workspace Dimensions: {df.shape[0]} active vehicles mapped.")


print("\n=========================================================")
print("=== PHASE 2: AUTOMOTIVE LOGISTICS QUANTIFICATIONS     ===")
print("=========================================================")

# --- TASK A: MANUFACTURER PRICING POSITIONING ---
make_summary = df.groupby('make').agg({
    'price': 'mean',
    'horsepower': 'mean',
    'city_mpg': 'mean'
}).sort_values(by='price', ascending=False)

print("Top 5 Manufacturer Brand Positioning Matrix (Average Benchmarks):")
print(make_summary.head(5))

# --- TASK B: COVARIANCE SPACE CORRELATION MAPPING ---
analytical_features = ['price', 'horsepower', 'curb_weight', 'wheel_base', 'length', 'city_mpg', 'highway_mpg']
corr_matrix = df[analytical_features].corr()
print("\nCore Pearson Correlation Spaces Coefficients Matrix:")
print(corr_matrix)


print("\n=========================================================")
print("=== PHASE 3: ISOLATED METRIC PIPELINE EXTRACTOR          ===")
print("=========================================================")

# --- PLOT 1: Box Plot (Price Distribution Across Fuel Powertrains) ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='fuel_type', y='price', hue='fuel_type', palette='Set2', legend=False)
plt.title('Vehicle Valuation Matrix Spread by Fuel Configuration', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Fuel Type Configuration')
plt.ylabel('Gross Transaction Price ($)')
plt.tight_layout()
plt.savefig("output/chart1_price_fuel_boxplot.png", dpi=300)
plt.close()

# --- PLOT 2: Scatter Plot (Horsepower vs Valuation Elasticity) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='horsepower', y='price', hue='Brand_Tier', alpha=0.7, palette='deep', edgecolor='none')
plt.title('Engine Horsepower Capacity vs Transaction Price Elasticity', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Horsepower (HP)')
plt.ylabel('Price ($)')
plt.tight_layout()
plt.savefig("output/chart2_horsepower_scatter.png", dpi=300)
plt.close()

# --- PLOT 3: Heatmap Matrix (Core Feature Association Spaces Map) ---
plt.figure(figsize=(9, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, cbar=False)
plt.title('Automobile Dimensional Covariance & Correlation Matrix Map', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("output/chart3_correlation_matrix.png", dpi=300)
plt.close()

# --- PLOT 4: Bar Plot (Top Active Brands Valuation Leaderboard) ---
plt.figure(figsize=(12, 6))
# Select top 10 brands by volume to keep the horizontal chart perfectly readable
top_makes = df['make'].value_counts().head(10).index
df_top_makes = df[df['make'].isin(top_makes)]
make_order = df_top_makes.groupby('make')['price'].mean().sort_values(ascending=False).index

sns.barplot(data=df_top_makes, x='price', y='make', hue='make', order=make_order, palette='viridis', errorbar=None, legend=False)
plt.title('Top 10 Manufacturer Brand Positioning Matrix (Average Price)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Average Price ($)')
plt.ylabel('Vehicle Make')
plt.tight_layout()
plt.savefig("output/chart4_brand_price_barplot.png", dpi=300)
plt.close()



print("\n=========================================================")
print("=== PHASE 4: COMPILING MASTER COMBINED EXECUTIVE BOARD ===")
print("=========================================================")

fig, axes = plt.subplots(2, 2, figsize=(22, 16))

# Panel 1: Fuel Type Pricing Breakdown (Box Plot Layout)
sns.boxplot(data=df, x='fuel_type', y='price', hue='fuel_type', palette='Set2', ax=axes[0, 0], legend=False)
axes[0, 0].set_title('Price Distribution by Fuel Powertrain Configurations', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Fuel Type')
axes[0, 0].set_ylabel('Price ($)')

# Panel 2: Engine Elasticity vs Valuation (Scatter Layout)
sns.scatterplot(data=df, x='horsepower', y='price', hue='Brand_Tier', alpha=0.6, palette='deep', ax=axes[0, 1])
axes[0, 1].set_title('Horsepower Scale Curves vs Price Slices by Tier', fontsize=13, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Engine Horsepower (HP)')
axes[0, 1].set_ylabel('Price ($)')

# Panel 3: Dimensional Metrics Covariances (Heatmap Layout)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, ax=axes[1, 0], cbar=False)
axes[1, 0].set_title('Automotive Technical Feature Correlation Matrix Map', fontsize=13, fontweight='bold', pad=15)

# Panel 4: Top Manufacturer Pricing Brackets (Bar Layout)
sns.barplot(data=df_top_makes, x='price', y='make', hue='make', order=make_order, palette='viridis', ax=axes[1, 1], errorbar=None, legend=False)
axes[1, 1].set_title('Average Market Price Across Top Volume Manufacturers', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Average Transaction Price ($)')
axes[1, 1].set_ylabel('')

plt.tight_layout()
plt.savefig("output/automobile_executive_dashboard.png", dpi=300)
plt.close()


print("\n=========================================================")
print("=== PHASE 5: BUSINESS DATA STORYTELLING & STRATEGY     ===")
print("=========================================================")
hp_price_corr = corr_matrix.loc['horsepower', 'price']
weight_price_corr = corr_matrix.loc['curb_weight', 'price']
mpg_price_corr = corr_matrix.loc['city_mpg', 'price']

print(f"1. THE PERFORMANCE ASSET PREMIUM:")
print(f"   - Pearson Correlation (Horsepower vs Price): {hp_price_corr:.4f}")
print(f"   - Pearson Correlation (Curb Weight vs Price): {weight_price_corr:.4f}")
print("   Corporate Narrative: Pricing matrices are heavily driven by engineering metrics.")
print("   Heavy engineering metrics (`curb_weight`) and power outputs (`horsepower`) display strong positive correlations with price.")
print("2. CONSUMER ELASTICITY INSIGHT:")
print(f"   - Pearson Correlation (City MPG vs Price): {mpg_price_corr:.4f}")
print("   Strategic Verdict: Fuel efficiency parameters hold a strong negative correlation coefficient against car values.")
print("   Premium market segments value horsepower and structural weight far above economy consumption metrics, allowing luxury marques to capture higher pricing margins on high-performance models.")