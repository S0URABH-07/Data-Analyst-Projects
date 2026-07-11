import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

var =  pd.read_csv("data/covid_data.csv")
print(var.isnull().sum())

var['Date'] = pd.to_datetime(var['Date'])

var['Death_Rate_%'] = np.where(var['Confirmed'] > 0, (var['Deaths'] / var['Confirmed']) * 100, 0)
var['Recovery_Rate_%'] = np.where(var['Confirmed'] > 0, (var['Recovered'] / var['Confirmed']) * 100, 0)

# 1. Total Cumulative Cases by Country (Taking the latest date records)
latest_date = var['Date'].max()
latest_var = var[var['Date'] == latest_date]

country_summary = latest_var.groupby('Country/Region').agg({
    'Confirmed': 'sum',
    'Deaths': 'sum',
    'Recovered': 'sum'
}).sort_values(by='Confirmed', ascending=False)

print("\nTop 5 Affected Countries (Latest Cumulative Metrics):")
print(country_summary.head(5))

# 2. Monthly Trend Aggregation
# Extract Year-Month directly into a string formatting for clean plotting
var['Year_Month'] = var['Date'].dt.strftime('%Y-%m')
monthly_trend = var.groupby('Year_Month').agg({
    'New cases': 'sum',
    'New deaths': 'sum',
    'Confirmed': 'sum'
}).sort_index()

print("\nMonthly Growth Summary Trends:")
print(monthly_trend.head(5))


print("\n===EXPLORATORY CORRELATIONS ===")
numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths']
correlation_matrix = latest_var[numeric_cols].corr()

print("\nCorrelation Matrix Table:")
print(correlation_matrix)


print("\n===GENERATING & SAVING METRIC DASHBOARD ===")
plt.figure(figsize=(18, 5))

# Line Plot (Global Cumulative Metrics Trend) ---
plt.subplot(1, 3, 1)
global_daily = var.groupby('Date')[['Confirmed', 'Active']].sum()
plt.plot(global_daily.index, global_daily['Confirmed'], color='darkorange', label='Total Confirmed', linewidth=2)
plt.plot(global_daily.index, global_daily['Active'], color='dodgerblue', label='Active Cases', linestyle='--')
plt.title('Global COVID-19 Timeline Trend')
plt.xlabel('Timeline')
plt.ylabel('Cases Count')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()

# Area Plot (Monthly New Cases Volume) ---
plt.subplot(1, 3, 2)
months_labels = monthly_trend.index.astype(str)
plt.fill_between(months_labels, monthly_trend['New cases'], color='crimson', alpha=0.4, label='New Cases Volume')
plt.plot(months_labels, monthly_trend['New cases'], color='crimson', linewidth=1.5)
plt.title('Monthly New Infections Volume')
plt.xlabel('Year-Month')
plt.ylabel('New Incident Count')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()

# Heatmap (Correlation Space Mapping) ---
plt.subplot(1, 3, 3)
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
plt.title('EDA Feature Correlation Heatmap')
plt.colorbar()

ticks = np.arange(len(correlation_matrix.columns))
plt.xticks(ticks, correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(ticks, correlation_matrix.columns)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        val = correlation_matrix.iloc[i, j]
        plt.text(i, j, f"{val:.2f}", ha='center', va='center', 
                 color='black' if abs(val) < 0.7 else 'white', fontweight='bold')

plt.tight_layout()

if not os.path.exists("output"):
    os.makedirs("output")
    
plt.savefig("output/covid_analysis_dashboard.png", dpi=300)

plt.show()