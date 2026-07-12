import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

var = pd.read_csv("data/Superstore.csv" , encoding="latin-1")
print(var.isnull().sum())
print(var.shape)
var.columns = var.columns.str.replace(" ","_").str.replace("-","_")

# Handle Data Types: Parse operational date strings into real datetime objects
var['Order_Date'] = pd.to_datetime(var['Order_Date'])
var['Ship_Date'] = pd.to_datetime(var['Ship_Date'])

# Feature Engineering: Add strategic tracking columns
var['Year_Month'] = var['Order_Date'].dt.to_period('M')
var['Shipping_Duration_Days'] = (var['Ship_Date'] - var['Order_Date']).dt.days

print("\nIngestion Complete. Dataset Dimensions:", var.shape)

# Aggregate monthly numbers chronologically to view market cycles
monthly_metrics = var.groupby('Year_Month').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).sort_index()

print("\nMonthly Growth Velocity Trends (First 5 Months):")
print(monthly_metrics.head(5))


# PRICING ELASTICITY & DISCOUNT AUDIT
# Are deep markdown strategies eroding margins?
discount_impact = var.groupby('Discount').agg({
    'Profit': 'mean',
    'Sales': 'mean',
    'Order_ID': 'count'
}).rename(columns={'Order_ID': 'Transaction_Count'})

print("\nDiscount Performance Pricing Profile Matrix:")
print(discount_impact)


# MARKET SEGMENTATION DATA STORYTELLING
# Quantify customer lifetime value clusters across macro profiles
segment_summary = var.groupby('Segment').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Customer_ID': 'nunique'
}).rename(columns={'Customer_ID': 'Unique_Customers'})

segment_summary['Revenue_Per_Customer'] = segment_summary['Sales'] / segment_summary['Unique_Customers']
segment_summary['Profit_Margin_%'] = (segment_summary['Profit'] / segment_summary['Sales']) * 100
print("\nStrategic Market Segment Matrix Summary:")
print(segment_summary)


# PRODUCT PERFORMANCE LEADERBOARD & LOSS-LEADERS
# Unmask sub-categories that are bleeding actual cash flow out of operations
subcategory_performance = var.groupby('Sub_Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).sort_values(by='Profit', ascending=False)

print("\nTop 3 Most Profitable Sub-Categories:")
print(subcategory_performance.head(3))

print("\nTop 3 Underperforming Cash-Bleeding Sub-Categories:")
print(subcategory_performance.tail(3))


# Set clean corporate visual theme formatting styles
sns.set_theme(style="white", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'  # rcParams = Runtime Configuration Parameters

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# Line Chart (Financial Progress Horizons Timeline)
months_str = monthly_metrics.index.astype(str)
sns.lineplot(x=months_str, y=monthly_metrics['Sales'], ax=axes[0, 0], color='#1f77b4', linewidth=2.5, label='Gross Sales')
sns.lineplot(x=months_str, y=monthly_metrics['Profit'], ax=axes[0, 0], color='#2ca02c', linewidth=2, linestyle='--', label='Net Profit')
axes[0, 0].set_title('Corporate Financial Performance Velocity Timeline', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Timeline Horizon')
axes[0, 0].set_ylabel('Capital Volume ($)')
# Filter tick steps frequency to keep the labels from crashing into each other
for index, label in enumerate(axes[0, 0].get_xticklabels()):
    if index % 6 != 0:
        label.set_visible(False)
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].legend()

# Horizontal Bar Plot (Loss-Leader Financial Audit)
sns.barplot(
    data=subcategory_performance.reset_index(), 
    x='Profit', 
    y='Sub_Category', 
    ax=axes[0, 1], 
    palette='coolwarm_r', 
    edgecolor='black', 
    hue='Sub_Category', 
    legend=False
)
axes[0, 1].axvline(0, color='red', linestyle='-', linewidth=1.5, alpha=0.7)
axes[0, 1].set_title('Product Profitability Leakage Analysis by Sub-Category', fontsize=13, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Net Capital Profit Contribution ($)')
axes[0, 1].set_ylabel('Product Sub-Category Class')

# Heatmap Matrix (Core Operational Metric Associations)
numerical_features = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping_Duration_Days']
correlation_space = var[numerical_features].corr()
sns.heatmap(
    correlation_space, 
    annot=True, 
    cmap='vlag', 
    fmt=".3f", 
    vmin=-1, 
    vmax=1, 
    square=True, 
    ax=axes[1, 0], 
    cbar_kws={'label': 'Correlation Strength Coefficient'}
)
axes[1, 0].set_title('Operational Efficiency Matrix Feature Correlations', fontsize=13, fontweight='bold', pad=15)

# Segment Profile Comparative Distribution Breakdowns
segment_metrics = var.groupby(['Category', 'Segment'])['Sales'].sum().reset_index()
sns.barplot(data=segment_metrics, x='Sales', y='Category', hue='Segment', palette='Blues_d', ax=axes[1, 1], edgecolor='black')
axes[1, 1].set_title('Hierarchical Market Category Share by Customer Segment', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Gross Sales Volume ($)')
axes[1, 1].set_ylabel('Core Business Category')

plt.tight_layout()

if not os.path.exists("output"):
    os.makedirs("output")
plt.savefig("output/superstore_executive_dashboard.png", dpi=300, bbox_inches='tight')

plt.show()

print("\n=========================================================")
print("=== EXECUTIVE CORE BUSINESS DATA STORYTELLING ===")
print("=========================================================")
print("1. THE PRICING ELASTICITY TRAP:")
print(f"   - Average profit per transaction at 0% Discount: ${var[var['Discount'] == 0]['Profit'].mean():.2f}")
print(f"   - Average profit per transaction at 20% Discount: ${var[var['Discount'] == 0.2]['Profit'].mean():.2f}")
print(f"   - Average profit per transaction at 50%+ Discount: ${var[var['Discount'] >= 0.5]['Profit'].mean():.2f}")
print("   Strategic Verdict: High promotional discounting does NOT scale sales value—it decimates margins.")
print("2. OPERATIONS INVENTORY TARGETING VULNERABILITY:")
lowest_sub = subcategory_performance.idxmin()['Profit']
print(f"   - The underperforming category is '{lowest_sub}', leaking serious operating cash flow.")
print(f"   Strategic Actionable: Immediately target '{lowest_sub}' procurement margins or test price increases.")