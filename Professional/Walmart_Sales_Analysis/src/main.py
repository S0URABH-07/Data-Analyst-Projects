import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

df = pd.read_csv("data/walmart_sales.csv")

# Ensure target storage folders exist cleanly
if not os.path.exists("output"):
    os.makedirs("output")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
# If there are parsing anomalies, fallback gracefully
if df['Date'].isna().sum() > 0:
    df['Date'] = pd.to_datetime(pd.read_csv("data/walmart_sales.csv")['Date'], errors='coerce')

# EDA 
print(f"[EDA] Data Profile: {df.shape[0]} weekly regional records mapped.")
print(f"Missing Values across the whole matrix: {df.isnull().sum().sum()} gaps found.")


df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Year_Month_Str'] = df['Date'].dt.strftime('%Y-%m')

df['Estimated_Net_Profit'] = df['Weekly_Sales'] * 0.085

store_volume = df.groupby('Store')['Weekly_Sales'].transform('sum')
df['Store_Performance_Tier'] = pd.qcut(store_volume, q=3, labels=['Tier 3 - Low Volume', 'Tier 2 - Mid Volume', 'Tier 1 - Enterprise Hub'])

print("[Ingestion Framework Completed Successfully]")


# --- TASK A: GLOBAL PORTFOLIO MARGIN QUANTIFICATION ---
total_sales = df['Weekly_Sales'].sum()
total_profit = df['Estimated_Net_Profit'].sum()
print(f"Total Sales Revenue Tracked: ${total_sales:,.2f}")
print(f"Simulated Corporate Net Profit Margin: ${total_profit:,.2f}")

# --- TASK B: HOLIDAY SPIKE COEFFICIENTS ---
holiday_performance = df.groupby('Holiday_Flag')['Weekly_Sales'].agg(['mean', 'median', 'std'])
holiday_performance.index = ['Standard Work Week', 'Promotional Holiday Week']
print("\nHoliday Demand Amplification Profile Matrix:")
print(holiday_performance)

# --- TASK C: MACRO CUSTOMER SEGMENT TIERS AUDIT ---
tier_summary = df.groupby('Store_Performance_Tier', observed=False).agg({
    'Weekly_Sales': 'mean',
    'CPI': 'mean',
    'Unemployment': 'mean',
    'Store': 'nunique'
}).rename(columns={'Weekly_Sales': 'Avg_Weekly_Sales', 'Store': 'Store_Count'})
print("\nStrategic Performance Tiers Matrix Summary:")
print(tier_summary)


# Line Plot 
plt.figure(figsize=(12, 6))
timeline_metrics = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
plt.plot(timeline_metrics['Date'], timeline_metrics['Weekly_Sales'], color='#0071ce', linewidth=2.5, label='Gross Weekly Revenue')
# Highlight structural holiday markers
holiday_dates = df[df['Holiday_Flag'] == 1]['Date'].unique()
plt.vlines(holiday_dates, ymin=timeline_metrics['Weekly_Sales'].min(), ymax=timeline_metrics['Weekly_Sales'].max(), 
           color='#ffc220', alpha=0.3, linestyle='--', label='Holiday Promotional Target')
plt.title('Chronological Retail Sales Performance & Holiday Peak Timeline', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Timeline Horizon')
plt.ylabel('Aggregate Weekly Revenue ($)')
plt.legend()
plt.tight_layout()
plt.savefig("output/chart1_sales_timeline.png", dpi=300)
plt.close()

# Heatmap Matrix (
plt.figure(figsize=(10, 9))
economic_features = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag', 'Estimated_Net_Profit']
correlation_space = df[economic_features].corr()
sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation Coefficient Strength'})
plt.title('Retail Feature Covariance & Macro-Economic Correlation Spaces Map', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("output/chart2_economic_correlation_matrix.png", dpi=300)
plt.close()

# Scatter Plot 
plt.figure(figsize=(11, 6))
sns.scatterplot(data=df, x='Unemployment', y='Weekly_Sales', hue='Store_Performance_Tier', alpha=0.4, palette='deep', edgecolor='none')
plt.title('Market Elasticity Study: Prevailing Regional Unemployment Index vs Weekly Store Revenue', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Prevailing Unemployment Rate (%)')
plt.ylabel('Weekly Store Sales ($)')
plt.tight_layout()
plt.savefig("output/chart3_unemployment_vs_sales_scatter.png", dpi=300)
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(22, 16))

axes[0, 0].plot(timeline_metrics['Date'], timeline_metrics['Weekly_Sales'], color='#0071ce', linewidth=2)
axes[0, 0].set_title('Chronological Retail Revenue Horizon Velocity', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Timeline Horizon')
axes[0, 0].set_ylabel('Total Sales Revenue ($)')


sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Retail Metric & Economic Correlation Spaces Map', fontsize=13, fontweight='bold', pad=15)


sns.scatterplot(data=df, x='Unemployment', y='Weekly_Sales', hue='Store_Performance_Tier', alpha=0.2, palette='deep', ax=axes[1, 0], legend=False)
axes[1, 0].set_title('Macro Unemployment Index Elasticity Thresholds', fontsize=13, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('Unemployment Rate (%)')
axes[1, 0].set_ylabel('Weekly Store Sales ($)')


sns.boxplot(data=df, x='Store_Performance_Tier', y='Estimated_Net_Profit', hue='Store_Performance_Tier', palette='Set2', ax=axes[1, 1], width=0.5, legend=False)
axes[1, 1].set_title('Calculated Net Profit Variance Slices by Store Tier Class', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Stratified Performance Tiers')
axes[1, 1].set_ylabel('Estimated Weekly Net Profit ($)')

plt.tight_layout()
plt.savefig("output/walmart_executive_master_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()


print("\n=========================================================")
print("===       BUSINESS DATA STORYTELLING & STRATEGY        ===")
print("=========================================================")
print("1. THE HOLIDAY AMPLIFICATION INSIGHT:")
holiday_mean = holiday_performance.loc['Promotional Holiday Week', 'mean']
standard_mean = holiday_performance.loc['Standard Work Week', 'mean']
premium_ratio = ((holiday_mean - standard_mean) / standard_mean) * 100
print(f"   - Average Promotional Holiday Week Revenue: ${holiday_mean:,.2f}")
print(f"   - Average Standard Performance Week Revenue: ${standard_mean:,.2f}")
print(f"   Corporate Narrative: Promotional holiday events trigger a massive {premium_ratio:.2f}% spike in weekly operations revenue.")
print("   Strategic Actionable: Supply chain distribution centers must scale up logistics buffers 14 days prior to these high-velocity blocks to avoid expensive stockouts.")
print("2. ECONOMIC ELASTICITY INSIGHT:")
print(f"   - Pearson Correlation (Unemployment vs Weekly Sales): {correlation_space.loc['Unemployment', 'Weekly_Sales']:.4f}")
print("   Strategic Verdict: The weak negative correlation coefficient signals high resilience against macro recessions.")
print("   Walmart functions as a defensive consumer-staple anchor; revenue remains stable even when local regional unemployment figures slide upward.")