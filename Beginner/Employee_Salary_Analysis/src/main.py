import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os

var = pd.read_csv("data/employees.csv").head(100)
print(var.columns.to_list())

print("MISSING VALUE COUNT PER COLUMN")
print(var.isnull().sum())

var["Gender"] = var["Gender"].fillna("Not Specified")
var["First Name"] = var["First Name"].fillna("Unknown")
var["Team"] = var["Team"].fillna("Unknown Team")
var["Senior Management"] = var["Senior Management"].fillna("Not Fixed")

print("After Fill Missing Value")
print(var.isnull().sum())

# Highest Salary with name
highest_salary = var["Salary"].max()
top_earner_index = var["Salary"].idxmax()
top_earner_name = var.loc[top_earner_index,"First Name"]
print(f"Highest Salary {highest_salary} scored by {top_earner_name}")

# Average Salary
average_salary = var["Salary"].mean()
print(f"Average salary {average_salary}")

# Department Wise Salary
team_total_salaries = var.groupby("Team")["Salary"].sum()
team_average_salaries = var.groupby("Team")["Salary"].mean()
print(f"Team wise Total salary {team_total_salaries}")
print(f"Team wise average salary {team_average_salaries.round(2)}")

# Gender Wise Salary
gender_total_salary = var.groupby("Gender")["Salary"].sum()
gender_avg_salary = var.groupby("Gender")["Salary"].mean()
print(f"Gender wise total salary {gender_total_salary}")
print(f"Gender wise Average Salary {gender_avg_salary.round(2)}")

plt.figure(figsize=(16, 5))

# Histogram 
plt.subplot(1, 3, 1)
plt.hist(var['Salary'], bins=12, color='teal', edgecolor='black')
plt.axvline(average_salary, color='red', linestyle='dashed', linewidth=2, label=f'Avg: ${average_salary:,.0f}')
plt.title('Overall Salary Distribution')
plt.xlabel('Salary ($)')
plt.ylabel('Number of Employees')
plt.legend()

# Box Plot
plt.subplot(1, 3, 2)
gender_labels = var['Gender'].unique()
gender_groups = []

for label in gender_labels:
    gender_groups.append(var[var['Gender'] == label]['Salary'])

plt.boxplot(gender_groups, labels=gender_labels)
plt.title('Salary Spread by Gender')
plt.ylabel('Salary ($)')

# Violin Plot
plt.subplot(1, 3, 3)
top_teams = var["Team"].value_counts().head(3).index
team_groups = []

for team in top_teams:
    team_groups.append(var[var['Team'] == team]['Salary'])

plt.violinplot(team_groups)
plt.xticks(range(1, len(top_teams) + 1), top_teams, rotation=15)
plt.title('Salary Density across Top Teams')
plt.ylabel('Salary ($)')

plt.tight_layout()

if not os.path.exists("output"):
    os.makedirs("output")
    
plt.savefig("output/employee_salary_dashboard.png", dpi=300)
print("--- Image Saved Successfully ---")
plt.show()