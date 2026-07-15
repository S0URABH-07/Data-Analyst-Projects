import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Establish corporate executive visual styles
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'


var = pd.read_csv("data/zomato.csv", encoding="latin-1")

var.columns = var.columns.str.strip()

if not os.path.exists("output"):
    os.makedirs("output")

print(f"[EDA] Data Profile: {var.shape[0]} restaurant entries mapped.")
print(f"[EDA] Missing Values: {var.isnull().sum().sum()} empty entries found.")

var['Cuisines'] = var['Cuisines'].fillna('Unknown')

# 1. Filter for a Uniform Currency Area (Indian Rupees) for stable cost modeling
var_inr = var[var['Currency'] == 'Indian Rupees(Rs.)'].copy()

median_cost = var_inr['Average Cost for two'].median()
var_inr['Average Cost for two'] = var_inr['Average Cost for two'].replace(0, median_cost)

# 2. Score Efficiency Rating Index
var_inr['Engagement_Score'] = var_inr['Aggregate rating'] * var_inr['Votes']

# 3. Categorical Segments
var_inr['Dining_Price_Tier'] = pd.cut(var_inr['Average Cost for two'], 
                                     bins=[0, 400, 1000, 2500, 100000], 
                                     labels=['Budget Dining', 'Casual Dining', 'Premium Dining', 'Luxury Dining'])

print(f"[Ingestion Complete] Indian Sub-dataset Dimensions: {var_inr.shape[0]} active locations mapped.")


# TASK A: ONLINE DELIVERY REVENUE ADVANTAGE 
delivery_comparison = var_inr.groupby('Has Online delivery').agg({
    'Aggregate rating': 'mean',
    'Votes': 'mean',
    'Average Cost for two': 'mean'
}).rename(columns={'Aggregate rating': 'Avg_Rating', 'Votes': 'Avg_Votes', 'Average Cost for two': 'Avg_Cost'})

print("🚚 Delivery Integration Performance Matrix:")
print(delivery_comparison)

# TASK B: IDENTIFYING BEST PERFORMING RESTAURANTS
best_restaurants = var_inr[var_inr['Aggregate rating'] >= 4.5].sort_values(by='Votes', ascending=False)
print("\n🏆 Top 5 Best & Most Engaged Restaurants:")
print(best_restaurants[['Restaurant Name', 'Aggregate rating', 'Votes', 'Average Cost for two']].head(5).to_string(index=False))


# Box Plot 
plt.figure(figsize=(10, 6))
sns.boxplot(data=var_inr, x='Dining_Price_Tier', y='Average Cost for two', hue='Has Online delivery', palette='Set2', width=0.6)
plt.title('Restaurant Pricing Tiers & Online Delivery Availability', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Dining Tier Class')
plt.ylabel('Average Cost for Two (INR)')
plt.yscale('log')  
plt.tight_layout()
plt.savefig("output/chart1_pricing_tiers_boxplot.png", dpi=300)
plt.close()

# KDE Distribution Plot 
plt.figure(figsize=(10, 6))
sns.kdeplot(data=var_inr, x='Aggregate rating', hue='Has Online delivery', fill=True, common_norm=False, palette='coolwarm', alpha=0.5, linewidth=2)
plt.title('Aggregate Rating Curve Density Split by Delivery Integration', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Aggregate Customer Rating (out of 5.0)')
plt.ylabel('Density Distribution Weight')
plt.tight_layout()
plt.savefig("output/chart2_ratings_kde_plot.png", dpi=300)
plt.close()

# Heatmap Matrix 
plt.figure(figsize=(10, 9))
analytical_features = ['Average Cost for two', 'Price range', 'Aggregate rating', 'Votes', 'Engagement_Score']
correlation_space = var_inr[analytical_features].corr()
sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation strength'})
plt.title('Zomato Feature Covariance & Operational Correlation Space', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("output/chart3_correlation_matrix.png", dpi=300)
plt.close()

# Scatter Plot 
plt.figure(figsize=(11, 6))
sns.scatterplot(data=var_inr, x='Votes', y='Aggregate rating', hue='Dining_Price_Tier', alpha=0.5, palette='deep', edgecolor='none')
plt.title('Audience Interaction: Customer Rating Scores vs Votes Volume', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Customer Votes Volume')
plt.ylabel('Aggregate Rating Score')
plt.tight_layout()
plt.savefig("output/chart4_votes_vs_ratings_scatter.png", dpi=300)
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(22, 16))

sns.kdeplot(data=var_inr, x='Aggregate rating', hue='Has Online delivery', fill=True, common_norm=False, palette='coolwarm', alpha=0.5, linewidth=2, ax=axes[0, 0])
axes[0, 0].set_title('Aggregate Rating Density Curve by Online Delivery Status', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Aggregate Rating Score')

sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Zomato Metric Correlation & Asset Covariances Map', fontsize=13, fontweight='bold', pad=15)

sns.boxplot(data=var_inr, x='Dining_Price_Tier', y='Average Cost for two', hue='Dining_Price_Tier', palette='Set2', ax=axes[1, 0], width=0.5, legend=False)
axes[1, 0].set_title('Customer Cost Distributions across Dining Price Tiers', fontsize=13, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('Dining Price Tiers')
axes[1, 0].set_ylabel('Average Cost for Two (INR - Log Scale)')
axes[1, 0].set_yscale('log')

sns.countplot(data=var_inr, x='Price range', hue='Has Online delivery', palette='viridis', ax=axes[1, 1], edgecolor='black')
axes[1, 1].set_title('Delivery Service Availability across Standard Price Ranges', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Price Range Tier (1-4)')
axes[1, 1].set_ylabel('Restaurant Count')

plt.tight_layout()
plt.savefig("output/zomato_executive_master_dashboard.png", dpi=300, bbox_inches='tight')
print(" -> [Saved Executive Master Dashboard] -> 'output/zomato_executive_master_dashboard.png'")
plt.close()


print("\n=========================================================")
print("===       BUSINESS DATA STORYTELLING & STRATEGY       ===")
print("=========================================================")
print("1. THE DELIVERY INTEGRATION CONVERSION IMPACT:")
delivery_yes_rating = delivery_comparison.loc['Yes', 'Avg_Rating']
delivery_no_rating = delivery_comparison.loc['No', 'Avg_Rating']
rating_lift = ((delivery_yes_rating - delivery_no_rating) / delivery_no_rating) * 100
print(f"   - Average Rating for Restaurants WITH Delivery Option: {delivery_yes_rating:.2f}/5.0")
print(f"   - Average Rating for Restaurants WITHOUT Delivery Option: {delivery_no_rating:.2f}/5.0")
print(f"   Corporate Narrative: Integrating online delivery option correlates with a {rating_lift:.2f}% lift in aggregate user feedback score.")
print("2. RISK MITIGATION STRATEGY:")
print(f"   - Pearson Correlation (Average Cost vs Aggregate Rating): {correlation_space.loc['Average Cost for two', 'Aggregate rating']:.4f}")
print("   Strategic Verdict: A weak positive correlation coefficient signals that cost alone does not guarantee superior culinary execution.")
print("   Actionable Business Strategy: Platform algorithms should highlight highly-rated budget-friendly locations on the home screen feed to maximize customer retention and app interaction velocity.")