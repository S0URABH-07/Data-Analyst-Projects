import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

var = pd.read_csv("data/banking_customers.csv")
print(var.isnull().sum())
print(var.shape)

if not os.path.exists("output"):
    os.makedirs("output")

# Total networth 
var["Net_Worth"] = var["SAVINGS"] - var["DEBT"]

var['High_Risk_Gambling_Flag'] = np.where((var['CAT_GAMBLING'] == 'High') & (var['DEFAULT'] == 1), 1, 0)

print("[Ingestion Framework Completed Successfully]")


# CREDIT DEFAULT & RISK QUANTIFICATION 
total_defaults = var['DEFAULT'].sum()
default_rate = (total_defaults / var.shape[0]) * 100
print(f"Total Systemic Customer Defaults Mapped: {total_defaults}")
print(f"Portfolio Risk / Default Rate: {default_rate:.2f}%")

# INCOME & DEBT DISTRIBUTIONS BY RISK PROFILE
risk_income_profile = var.groupby('DEFAULT')[['INCOME', 'DEBT', 'SAVINGS']].mean()
print("\nAverage Core Assets/Liabilities by Default Status:")
print(risk_income_profile)

# CREDIT SCORE STRATIFICATION
var['Credit_Score_Group'] = pd.cut(var['CREDIT_SCORE'], bins=[300, 580, 670, 740, 850], 
                                  labels=['Poor (<580)', 'Fair (580-670)', 'Good (670-740)', 'Excellent (740+)'])
credit_group_default = var.groupby('Credit_Score_Group', observed=False)['DEFAULT'].mean() * 100
print("\nDefault Probability Across Credit Score Segments (%):")
print(credit_group_default)


# KDE Plot (Income & Savings Distribution Curve vs Default) 
plt.figure(figsize=(10, 6))
sns.kdeplot(data=var, x='INCOME', hue='DEFAULT', fill=True, common_norm=False, palette='crest', alpha=0.5, linewidth=2)
plt.title('Income Density Curve (KDE) Split by Account Default Status', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Annual Gross Income ($)')
plt.ylabel('Density Distribution Weight')
plt.tight_layout()
plt.savefig("output/chart1_income_kde_plot.png", dpi=300)
plt.close()

# KDE Plot (Credit Score Distribution Breakdown) 
plt.figure(figsize=(10, 6))
sns.kdeplot(data=var, x='CREDIT_SCORE', hue='DEFAULT', fill=True, common_norm=False, palette='magma', alpha=0.4, linewidth=2)
plt.title('Credit Score Density Horizon Curve (KDE) by Portfolio Risk', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Customer Credit Score')
plt.ylabel('Density Distribution Weight')
plt.tight_layout()
plt.savefig("output/chart2_credit_score_kde_plot.png", dpi=300)
plt.close()

# Heatmap Matrix (Correlation Space Mapping Grid)
plt.figure(figsize=(10, 9))
financial_features = ['INCOME', 'SAVINGS', 'DEBT', 'CREDIT_SCORE', 'R_DEBT_INCOME', 'DEFAULT']
correlation_space = var[financial_features].corr()
sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation strength'})
plt.title('Banking Metric Correlation Space Matrix Map', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("output/chart3_correlation_matrix.png", dpi=300)
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(22, 16))

# Panel 1: Income Density (KDE Plot)
sns.kdeplot(data=var, x='INCOME', hue='DEFAULT', fill=True, common_norm=False, palette='crest', alpha=0.5, linewidth=2, ax=axes[0, 0])
axes[0, 0].set_title('Income Density Curve Split by Default Status', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Annual Income ($)')

# Panel 2: Credit Score Demographics (KDE Plot)
sns.kdeplot(data=var, x='CREDIT_SCORE', hue='DEFAULT', fill=True, common_norm=False, palette='magma', alpha=0.4, linewidth=2, ax=axes[0, 1])
axes[0, 1].set_title('Customer Credit Score Density Profile Curve', fontsize=13, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Credit Score')

# Panel 3: Correlation Matrix Space (Heatmap)
sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True, ax=axes[1, 0], cbar=False)
axes[1, 0].set_title('Financial Feature Association Matrix Map', fontsize=13, fontweight='bold', pad=15)

# Panel 4: Credit Score Group Risk Exposure Breakdowns (Bar Plot)
sns.barplot(data=var, x='Credit_Score_Group', y='DEFAULT', hue='Credit_Score_Group', palette='Oranges_r', ax=axes[1, 1], errorbar=None, edgecolor='black', legend=False)
axes[1, 1].set_title('Portfolio Account Default Probability by Score Buckets', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Stratified Credit Score Classes')
axes[1, 1].set_ylabel('Probability of Default Rate (Ratio)')

plt.tight_layout()
plt.savefig("output/banking_executive_master_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n[Seaborn] Spawning complex multi-dimensional Pairplot grid canvas...")
pairplot_features = ['INCOME', 'SAVINGS', 'DEBT', 'CREDIT_SCORE', 'DEFAULT']
pairplot_fig = sns.pairplot(data=var[pairplot_features], hue='DEFAULT', palette='Set1', diag_kind='kde', height=3.0)
pairplot_fig.savefig("output/banking_customer_pairplot_matrix.png", dpi=300)
print(" -> [Saved Seaborn Pairplot Matrix]")

print("1. THE LIQUID ASSET CRITICALITY:")
zero_savings_default = var[var['SAVINGS'] == 0]['DEFAULT'].mean() * 100
positive_savings_default = var[var['SAVINGS'] > 0]['DEFAULT'].mean() * 100
print(f"   - Loan Default probability for customers with $0 Savings: {zero_savings_default:.2f}%")
print(f"   - Loan Default probability for customers with active cash buffers (> $0 Savings): {positive_savings_default:.2f}%")
print("   Corporate Storytelling: Liquid cash reserves are a far stronger protection indicator against defaults than income alone.")
print("2. RISK MITIGATION STRATEGY:")
print(f"   - Pearson Correlation (Credit Score vs Default Flag): {correlation_space.loc['CREDIT_SCORE', 'DEFAULT']:.4f}")
print("   Actionable Business Strategy: Underwriting algorithms should tighten authorization flags for anyone falling into the Poor/Fair score groups, as default probabilities spike massively within those segments.")