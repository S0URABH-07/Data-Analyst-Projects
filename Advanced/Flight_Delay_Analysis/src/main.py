import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

var = pd.read_csv("data/Airline_Delay_Cause.csv")

if not os.path.exists("output"):
    os.makedirs("output")

print(f"Data Ingested: {var.shape[0]} airport-airline monthly records.")
missing_cells = var.isnull().sum().sum()
print(f"Before:-{missing_cells}")

if missing_cells > 0:
    var = var.fillna(0)

print("After:- ",var.isnull().sum().sum())

var['Delay_Rate_%'] = np.where(var['arr_flights'] > 0, (var['arr_del15'] / var['arr_flights']) * 100, 0)

var['Cancellation_Rate_%'] = np.where(var['arr_flights'] > 0, (var['arr_cancelled'] / var['arr_flights']) * 100, 0)

var['Avg_Delay_Intensity_Mins'] = np.where(var['arr_del15'] > 0, var['arr_delay'] / var['arr_del15'], 0)

var['Date'] = pd.to_datetime(var[['year', 'month']].assign(day=1))
var['Year_Month_Str'] = var['Date'].dt.strftime('%Y-%m')

print("[Ingestion Framework Completed Successfully]")


# --- TASK A: CARRIER PERFORMANCE LEADERBOARD (Delay by Airline) ---
carrier_summary = var.groupby('carrier_name').agg({
    'arr_flights': 'sum',
    'arr_del15': 'sum',
    'arr_delay': 'sum'
}).reset_index()
carrier_summary['Carrier_Delay_Rate_%'] = (carrier_summary['arr_del15'] / carrier_summary['arr_flights']) * 100
carrier_summary = carrier_summary.sort_values(by='Carrier_Delay_Rate_%', ascending=False)

print("Top 5 Airlines with the Highest At-Arrival Delay Rates:")
print(carrier_summary[['carrier_name', 'arr_flights', 'Carrier_Delay_Rate_%']].head(5).to_string(index=False))

# --- TASK B: HUB CONTINGENCY MATRIX (Delay by Airport) ---
# Isolate major hubs managing a significant baseline volume of traffic (>5,000 seasonal operations)
hub_summary = var.groupby('airport').agg({
    'arr_flights': 'sum',
    'arr_del15': 'sum',
    'arr_delay': 'sum'
}).reset_index()
hub_summary['Hub_Delay_Rate_%'] = (hub_summary['arr_del15'] / hub_summary['arr_flights']) * 100
top_hubs = hub_summary[hub_summary['arr_flights'] > 5000].sort_values(by='Hub_Delay_Rate_%', ascending=False)

print("\nTop 5 High-Volume Airport Hubs with Highest Delay Rates:")
print(top_hubs[['airport', 'arr_flights', 'Hub_Delay_Rate_%']].head(5).to_string(index=False))

# --- TASK C: SEASONAL DISRUPTION CYCLES (Seasonal Delay) ---
seasonal_trends = var.groupby('month').agg({
    'Delay_Rate_%': 'mean',
    'Cancellation_Rate_%': 'mean',
    'weather_delay': 'sum'
})
print("\nSeasonal Delay Averages Matrix (By Month 1-12):")
print(seasonal_trends.head(4))


# Line Plot 
plt.figure(figsize=(12, 6))
monthly_timeline = var.groupby('Date')[['Delay_Rate_%', 'Cancellation_Rate_%']].mean()
plt.plot(monthly_timeline.index, monthly_timeline['Delay_Rate_%'], color='darkorange', marker='o', linewidth=2.5, label='Mean Delay Rate %')
plt.plot(monthly_timeline.index, monthly_timeline['Cancellation_Rate_%'], color='crimson', marker='s', linewidth=2, linestyle='--', label='Mean Cancellation Rate %')
plt.title('Chronological Aviation Disruption & Cancellation Timeline Trends', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Timeline Horizon')
plt.ylabel('Operational Disruption Metric (%)')
plt.legend()
plt.tight_layout()
plt.savefig("output/chart1_seasonal_timeline.png", dpi=300)
plt.close()

# Heatmap Matrix
plt.figure(figsize=(10, 9))
cause_features = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay', 'arr_cancelled']
correlation_space = var[cause_features].corr()
sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation Coefficient Strength'})
plt.title('Aviation Disruption Cause Structural Correlation Space Matrix Map', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("output/chart2_cause_correlation_heatmap.png", dpi=300)
plt.close()

# Scatter Plot 
plt.figure(figsize=(11, 6))
sns.scatterplot(data=var[var['arr_flights'] > 100], x='arr_flights', y='Delay_Rate_%', alpha=0.4, color='teal', edgecolor='none')
plt.title('Network Density Analysis: Flight Operations Volume vs Arrival Delay Propensity', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Total Scheduled Arrival Flights Volume')
plt.ylabel('Calculated Monthly At-Arrival Delay Rate (%)')
plt.xscale('log')  # Logarithmic scale handles highly crowded regional vs massive international flight arrays
plt.tight_layout()
plt.savefig("output/chart3_volume_vs_delay_scatter.png", dpi=300)
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(22, 16))

axes[0, 0].plot(monthly_timeline.index, monthly_timeline['Delay_Rate_%'], color='darkorange', marker='o', linewidth=2, label='Delay Rate')
axes[0, 0].plot(monthly_timeline.index, monthly_timeline['Cancellation_Rate_%'], color='crimson', marker='x', linestyle='--', label='Cancellation Rate')
axes[0, 0].set_title('Chronological Disruption Timelines Horizons', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Timeline Horizon')
axes[0, 0].set_ylabel('Disruption Magnitude (%)')
axes[0, 0].legend()

sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Operational Disruption Source Correlation Spaces Map', fontsize=13, fontweight='bold', pad=15)

sns.scatterplot(data=var[var['arr_flights'] > 100], x='arr_flights', y='Delay_Rate_%', alpha=0.3, color='teal', ax=axes[1, 0])
axes[1, 0].set_title('Operations Volume Saturated Densities vs Flight Disruption Propensity', fontsize=13, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('Total Scheduled Flights (Log Scale)')
axes[1, 0].set_ylabel('Delay Rate (%)')
axes[1, 0].set_xscale('log')

top_10_carriers = carrier_summary.head(10)
sns.barplot(data=top_10_carriers, x='Carrier_Delay_Rate_%', y='carrier_name', hue='carrier_name', palette='Reds_r', ax=axes[1, 1], edgecolor='black', legend=False)
axes[1, 1].set_title('Delay Conversion Index Across Top 10 Major Commercial Carriers', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Carrier Specific Delay Rate (%)')
axes[1, 1].set_ylabel('')

plt.tight_layout()
plt.savefig("output/aviation_logistics_executive_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()


print("\n=========================================================")
print("===        BUSINESS DATA STORYTELLING & STRATEGY       ===")
print("=========================================================")
print("1. THE LATE-AIRCRAFT PROPAGATION WAVE:")
late_aircraft_nas_corr = correlation_space.loc['late_aircraft_delay', 'carrier_delay']
print(f"   - Pearson Correlation Matrix Coefficient (Late Aircraft vs Carrier Delay): {late_aircraft_nas_corr:.4f}")
print("   Corporate Narrative: A high positive correlation indicates a systematic knock-on effect.")
print("   When an initial incoming flight is delayed due to airline issues (Carrier Delay), it triggers a rolling disruption.")
print("   The subsequent outbound leg is forced into a delayed departure because the aircraft arrived late (Late Aircraft Delay).")
print("2. RISK ACTIONABLE FOR LOGISTICS RESILIENCE:")
worst_carrier = carrier_summary.iloc[0]['carrier_name']
worst_carrier_rate = carrier_summary.iloc[0]['Carrier_Delay_Rate_%']
print(f"   - Operational Vulnerability Highlight: '{worst_carrier}' exhibits the weak baseline rating with a {worst_carrier_rate:.2f}% delay propensity.")
print("   Actionable Strategy: Target capacity allocation adjustments or audit ground-handling service SLA timelines for underperforming carriers to prevent these delays from spreading through primary hub networks.")