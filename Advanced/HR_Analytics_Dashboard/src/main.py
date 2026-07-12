import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

var = pd.read_csv("data/HR_Analytics.csv")
print(var.isnull().sum())

var['Attrition_Flag'] = np.where(var['Attrition'] == 'Yes', 1, 0)
var['OverTime_Flag'] = np.where(var['OverTime'] == 'Yes', 1, 0)
print(f"Ingestion Complete. Total Employees Mapped: {var.shape[0]} | Columns: {var.shape[1]}")

# CORE ATTRITION ANALYSIS 
total_attrition = var['Attrition_Flag'].sum()
attrition_rate = (total_attrition / var.shape[0]) * 100
print(f"Corporate Attrition Status: {total_attrition} employees left the company.")
print(f"Corporate Attrition Rate: {attrition_rate:.2f}%")

# ORGANIZATIONAL HEALTH PROFILE BY DEPARTMENT
dept_performance = var.groupby('Department').agg({
    'PerformanceRating': 'mean',
    'JobSatisfaction': 'mean',
    'Attrition_Flag': 'mean'
}).rename(columns={'Attrition_Flag': 'Attrition_Probability'})
dept_performance['Attrition_Probability'] *= 100

print("\nDepartmental Health Profiling Matrix:")
print(dept_performance)

# PROMOTION MATRIX & SYSTEMIC RETENTION DELAY 
promotion_impact = var.groupby('YearsSinceLastPromotion').agg({
    'Attrition_Flag': 'mean',
    'MonthlyIncome': 'mean',
    'EmployeeNumber': 'count'
}).rename(columns={'EmployeeNumber': 'Headcount', 'Attrition_Flag': 'Churn_Rate'}).head(6)
promotion_impact['Churn_Rate'] *= 100

print("\nPromotion Delay Operational Risk Profile Matrix:")
print(promotion_impact)

# Establish professional boardroom visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

if not os.path.exists("output"):
    os.makedirs("output")

fig, axes = plt.subplots(2,2,figsize=(20,14))

# Subplot 1: Attrition vs Department (Count Plot)
sns.countplot(data=var, x='Department', hue='Attrition', palette='Set2', edgecolor='black', ax=axes[0, 0])
axes[0, 0].set_title('Attrition Volume Structural Split across Departments', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Department Group')
axes[0, 0].set_ylabel('Employee Headcount')

# Subplot 2: Salary Distribution by Department & Attrition (Violin Plot)
sns.violinplot(data=var, x='MonthlyIncome', y='Department', hue='Attrition', split=True, palette='coolwarm', ax=axes[0, 1])
axes[0, 1].set_title('Salary Density & Attrition Spreads Across Departments', fontsize=13, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Monthly Income ($)')
axes[0, 1].set_ylabel('Department Group')

# Subplot 3: Correlation Space Matrix (Heatmap)
core_features = ['Attrition_Flag', 'MonthlyIncome', 'YearsSinceLastPromotion', 'Age', 'YearsAtCompany', 'OverTime_Flag']
correlation_space = var[core_features].corr()
sns.heatmap(correlation_space, annot=True, cmap='vlag', fmt=".3f", vmin=-1, vmax=1, square=True, ax=axes[1, 0], cbar_kws={'label': 'Correlation Coefficient'})
axes[1, 0].set_title('HR Metric Association Correlation Spaces', fontsize=13, fontweight='bold', pad=15)

# Subplot 4: Promotion Impact vs Attrition (Bar Plot)
sns.barplot(data=var, x='YearsSinceLastPromotion', y='Attrition_Flag', errorbar=None, color='#e74c3c', edgecolor='black', ax=axes[1, 1])
axes[1, 1].set_title('Attrition Propensity Over Promotion Stagnation Horizons', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Years Since Last Promotion Milestone')
axes[1, 1].set_ylabel('Probability of Leavers (Ratio)')

plt.tight_layout()
plt.savefig("output/hr_executive_master_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nSeparating individual structural figures out to storage...")

# Count Plot
plt.figure(figsize=(10, 6))
sns.countplot(data=var, x='Department', hue='Attrition', palette='Set2', edgecolor='black')
plt.title('Attrition Volume Structural Split across Departments', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Department Group')
plt.ylabel('Employee Headcount')
plt.tight_layout()
plt.savefig("output/chart1_attrition_countplot.png", dpi=300, bbox_inches='tight')
plt.close() # memory cleanup crew

# Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=var, x='MonthlyIncome', y='Department', hue='Attrition', split=True, palette='coolwarm')
plt.title('Salary Density & Attrition Spreads Across Departments', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Monthly Income ($)')
plt.ylabel('Department Group')
plt.tight_layout()
plt.savefig("output/chart2_salary_violinplot.png", dpi=300, bbox_inches='tight')
plt.close() # memory cleanup crew

# Heatmap Matrix
plt.figure(figsize=(9, 9))
sns.heatmap(correlation_space, annot=True, cmap='vlag', fmt=".3f", vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation Strength'})
plt.title('HR Metric Association Correlation Spaces', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("output/chart3_metric_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# Promotion Bar Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=var, x='YearsSinceLastPromotion', y='Attrition_Flag', errorbar=None, color='#e74c3c', edgecolor='black')
plt.title('Attrition Propensity Over Promotion Stagnation Horizons', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Years Since Last Promotion Milestone')
plt.ylabel('Probability of Leavers (Ratio)')
plt.tight_layout()
plt.savefig("output/chart4_promotion_barplot.png", dpi=300, bbox_inches='tight')
plt.close()


print("1. THE OVERTIME RETENTION IMPACT:")
overtime_attrition = var.groupby('OverTime')['Attrition_Flag'].mean() * 100
print(f"   - Attrition among employees working Overtime: {overtime_attrition['Yes']:.2f}%")
print(f"   - Attrition among employees NOT working Overtime: {overtime_attrition['No']:.2f}%")
print("   Corporate Storytelling: Employees logging overtime leave the company at more than twice the baseline rate.")
print("2. REVENUE VS RETENTION WINDOW:")
low_income_attrition = var[var['MonthlyIncome'] < 4000]['Attrition_Flag'].mean() * 100
print(f"   - Attrition Probability for entry-level compensation (<$4k/mo): {low_income_attrition:.2f}%")
print("   Strategic Actionable: Salary drops combined with promotion stagnation create high attrition risk zones. Target adjustments early.")