import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

var = pd.read_csv("data/athlete_events.csv")

print("Missing values per column before handling:")
print(var.isnull().sum())

# Manage Missing Value
var['Medal'] = var['Medal'].fillna('No Medal')
var['Age'] = var['Age'].fillna(var['Age'].mean())
var['Height'] = var['Height'].fillna(var['Height'].mean())
var['Weight'] = var['Weight'].fillna(var['Weight'].mean())

print(var.isnull().sum())

# BUSINESS & EDA QUESTIONS
# Country Ranking Most Gold Medals
gold_df = var[var['Medal'] == 'Gold']
country_gold_counts = gold_df.groupby('Team')['Medal'].count().sort_values(ascending=False)
print("\nTop 5 Countries with Most Gold Medals Historically:")
print(country_gold_counts.head(25))

# Athlete Analysis (Most Decorated Individual Competitors)
athlete_medals = var[var['Medal'] != 'No Medal'].groupby('Name')['Medal'].count().sort_values(ascending=False)
print("\nTop 5 Most Decorated Olympic Athletes (All Medals):")
print(athlete_medals.head(5))

# What is the average age of a gold medalist across all sports?
avg_gold_age = gold_df['Age'].mean()
median_gold_age = gold_df['Age'].median()
print(f"\nStrategic Talent Pipeline Discovery Insight:")
print(f"   - Average Age of Gold Medal Winner: {avg_gold_age:.1f} years old")
print(f"   - Median Age of Gold Medal Winner: {median_gold_age:.1f} years old")

var['Won_Medal_Flag'] = np.where(var['Medal'] != 'No Medal', 1, 0)

# Calculate the correlation matrix for numerical features
correlation_matrix = var[['Age', 'Height', 'Weight', 'Year', 'Won_Medal_Flag']].corr()
print("\nBiometric & Performance Correlation Matrix Table:")
print(correlation_matrix)

plt.figure(figsize=(18, 5))

# Histogram (Age Distribution of Gold Medalists)
plt.subplot(1, 3, 1)
plt.hist(gold_df['Age'], bins=15, color='#ffd700', edgecolor='black', alpha=0.8)
plt.axvline(avg_gold_age, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_gold_age:.1f} yrs')
plt.title('Age Distribution of Gold Medalists', fontsize=11, fontweight='bold')
plt.xlabel('Age of Athlete')
plt.ylabel('Gold Medals Count')
plt.legend()

# Heatmap (Biometric Feature Correlations Matrix)
plt.subplot(1, 3, 2)
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
plt.title('Biometric & Performance Correlation Map', fontsize=11, fontweight='bold')
plt.colorbar()

# Set up axis ticks natively
ticks = np.arange(len(correlation_matrix.columns))
plt.xticks(ticks, correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(ticks, correlation_matrix.columns)

# Write the correlation scores inside the heatmap grids
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        val = correlation_matrix.iloc[i, j]
        plt.text(i, j, f"{val:.2f}", ha='center', va='center', color='black', fontweight='bold')

# Bubble Plot (Height vs Weight across Sports) 
plt.subplot(1, 3, 3)
sport_summary = var.groupby('Sport').agg({
    'Height': 'mean',
    'Weight': 'mean',
    'ID': 'count' 
}).rename(columns={'ID': 'Athlete_Count'}).reset_index()

top_sports_bubble = sport_summary.sort_values(by='Athlete_Count', ascending=False).head(25)

plt.scatter(
    top_sports_bubble['Height'], 
    top_sports_bubble['Weight'], 
    s=top_sports_bubble['Athlete_Count'] * 0.08,
    color='purple', 
    alpha=0.6, 
    edgecolor='black'
)
plt.title('Sport Biometrics Bubble Chart', fontsize=11, fontweight='bold')
plt.xlabel('Average Height (cm)')
plt.ylabel('Average Weight (kg)')
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

if not os.path.exists("output"):
    os.makedirs("output")
    
plt.savefig("output/olympics_eda_dashboard.png", dpi=300)

plt.show()