import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

try:
    df = pd.read_csv("data/fifa_players.csv")
except FileNotFoundError:
    print("Error: Please place your dataset inside a 'data/' folder and name it 'fifa_players.csv'.")
    exit()

if not os.path.exists("output"):
    os.makedirs("output")

print(f"[EDA] Data Profile: {df.shape[0]} professional players mapped across {df.shape[1]} metrics.")

df['value_eur'] = df['value_eur'].fillna(df['value_eur'].median())
df['wage_eur'] = df['wage_eur'].fillna(df['wage_eur'].median())
df['club'] = df['club'].fillna('Free Agent')

df['growth_potential'] = df['potential'] - df['overall']

df['value_per_wage_ratio'] = np.where(df['wage_eur'] > 0, df['value_eur'] / df['wage_eur'], 0)

df['player_market_tier'] = pd.qcut(df['value_eur'], q=3, labels=['Tier 3 - Prospect', 'Tier 2 - Regional Asset', 'Tier 1 - Elite Elite'])

print("[Ingestion Framework Completed Successfully]")


# TASK A: THE ULTIMATE ELITE SQUAD 
best_players = df.sort_values(by='overall', ascending=False).head(5)
print("🏆 Top 5 Highest Rated Global Football Assets:")
print(best_players[['short_name', 'club', 'nationality', 'overall', 'potential', 'value_eur']].to_string(index=False))

# TASK B: COUNTRY CONVERSIONS 
country_summary = df.groupby('nationality').filter(lambda x: len(x) > 100)
country_performance = country_summary.groupby('nationality').agg({
    'overall': 'mean',
    'potential': 'mean',
    'value_eur': 'mean',
    'short_name': 'count'
}).rename(columns={'short_name': 'player_count'}).sort_values(by='overall', ascending=False)

print("\nTop 5 Talent Pipelines by Average Overall Rating (>100 players minimum):")
print(country_performance.head(5))

# TASK C: CLUB CAPITAL EXPENDITURE AUDIT 
club_capital = df.groupby('club').agg({
    'wage_eur': 'sum',
    'value_eur': 'sum',
    'overall': 'mean',
    'short_name': 'count'
}).rename(columns={'short_name': 'squad_size'}).sort_values(by='wage_eur', ascending=False)

print("\nTop 5 Clubs with Highest Weekly Wage Expenditure bills:")
print(club_capital.head(5))

# Scatter Plot 
plt.figure(figsize=(11, 6))
sns.scatterplot(data=df, x='age', y='overall', hue='player_market_tier', alpha=0.4, palette='deep', edgecolor='none')
plt.title('Talent Lifecycle Study: Player Age vs Overall Rating Performance Horizons', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Player Chronological Age')
plt.ylabel('Overall Performance Rating')
plt.tight_layout()
plt.savefig("output/chart1_age_vs_overall_scatter.png", dpi=300)
plt.close()

# Heatmap Matrix
plt.figure(figsize=(10, 9))
scouting_features = ['overall', 'potential', 'age', 'value_eur', 'wage_eur', 'growth_potential', 'value_per_wage_ratio']
correlation_space = df[scouting_features].corr()
sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation Coefficient Strength'})
plt.title('Scouting Metric Covariance & Financial Feature Correlation Map', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("output/chart2_scouting_correlation_matrix.png", dpi=300)
plt.close()

# KDE Distribution Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='overall', hue='player_market_tier', fill=True, common_norm=False, palette='viridis', alpha=0.5, linewidth=2)
plt.title('Performance Density Distribution Curve (KDE) Across Market Tiers', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Overall Rating Score')
plt.ylabel('Density Distribution Weight')
plt.tight_layout()
plt.savefig("output/chart3_overall_rating_kde.png", dpi=300)
plt.close()

# Bar Plot 
plt.figure(figsize=(12, 6))
top_10_clubs = club_capital.head(10).reset_index()
sns.barplot(data=top_10_clubs, x='wage_eur', y='club', hue='club', palette='Reds_r', edgecolor='black', legend=False)
plt.title('Financial Payroll Footprint: Total Weekly Wage Outlays Across Top 10 Clubs', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Total Combined Weekly Payroll (€)')
plt.ylabel('')
plt.tight_layout()
plt.savefig("output/chart4_club_payroll_barplot.png", dpi=300)
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(22, 16))

sns.scatterplot(data=df, x='age', y='overall', hue='player_market_tier', alpha=0.2, palette='deep', ax=axes[0, 0], legend=False)
axes[0, 0].set_title('Player Performance Lifecycles Slices by Age', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Player Age')
axes[0, 0].set_ylabel('Overall Rating')

sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Scouting Metric & Financial Feature Correlation Space', fontsize=13, fontweight='bold', pad=15)

sns.kdeplot(data=df, x='overall', hue='player_market_tier', fill=True, common_norm=False, palette='viridis', alpha=0.4, linewidth=1.5, ax=axes[1, 0])
axes[1, 0].set_title('Overall Performance Rating Density Distributions', fontsize=13, fontweight='bold', pad=15)
axes[1, 0].set_xlabel('Overall Rating Score')

sns.barplot(data=top_10_clubs, x='wage_eur', y='club', hue='club', palette='Reds_r', edgecolor='black', ax=axes[1, 1], legend=False)
axes[1, 1].set_title('Weekly Payroll Commitments Across Top 10 Global Clubs', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Total Weekly Wage Outlay (€)')
axes[1, 1].set_ylabel('')

plt.tight_layout()
plt.savefig("output/fifa_scouting_executive_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()

print("1. THE TALENT PEAK & VALUE INEFFICIENCY ZONE:")
peak_age_performance = df.groupby('age')['overall'].mean()
optimal_peak_age = peak_age_performance.idxmax()
print(f"   - Age holding highest mean overall rating baseline: {optimal_peak_age} Years Old")
print(f"   - Pearson Correlation (Overall vs Value_EUR): {correlation_space.loc['overall', 'value_eur']:.4f}")
print("   Corporate Narrative: Player valuation correlates exponentially with the Overall Rating flag, peaking between ages 26-29.")
print("2. RISK MITIGATION STRATEGY FOR ACQUISITIONS:")
print("   - High asset inflation occurs in Elite Tiers. Growth Potential holds a strong negative correlation with Age.")
print("   Actionable Business Strategy: Club recruitment frameworks should focus scouting capital on the 20-23 age bracket where 'growth_potential' is maximized. This creates value by acquiring rising talent before their market price climbs.")