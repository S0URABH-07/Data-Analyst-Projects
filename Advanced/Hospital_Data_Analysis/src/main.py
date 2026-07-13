import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Establish corporate executive visual styles
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'sans-serif'

var = pd.read_csv("data/hospital_patients.csv")
print(var.isnull().sum())

var.columns = var.columns.str.replace(" ","_").str.strip()

var['Date_of_Admission'] = pd.to_datetime(var['Date_of_Admission'])
var['Discharge_Date'] = pd.to_datetime(var['Discharge_Date'])

var['Length_of_Stay_Days'] = (var['Discharge_Date'] - var['Date_of_Admission']).dt.days

var['Length_of_Stay_Days'] = np.where(var['Length_of_Stay_Days'] < 0, 0, var['Length_of_Stay_Days'])

var['Cost_Per_Day'] = np.where(var['Length_of_Stay_Days'] > 0, 
                               var['Billing_Amount'] / var['Length_of_Stay_Days'], 
                               var['Billing_Amount'])

# 3. Numeric Risk Flag: Convert Test Results to simple ordinal rank values
# 'Normal' -> 0, 'Inconclusive' -> 1, 'Abnormal' -> 2
results_map = {'Normal': 0, 'Inconclusive': 1, 'Abnormal': 2}
var['Clinical_Risk_Flag'] = var['Test_Results'].map(results_map).fillna(1)

# --- TASK A: DISEASE EPIDEMIOLOGY DISTRIBUTION ---
condition_counts = var['Medical_Condition'].value_counts()
print("Disease Volume Distribution Leaderboard:")
print(condition_counts)

# --- TASK B: ADMISSION COST PROFILE BY INSURANCE ---
insurance_margins = var.groupby('Insurance_Provider')['Billing_Amount'].agg(['mean', 'median', 'sum'])
print("\nPayer Billing Concentrations Matrix:")
print(insurance_margins)

# --- TASK C: AGE PROFILE DEMOGRAPHIC PROFILE SEGMENTATION ---
var['Age_Group'] = pd.cut(var['Age'], bins=[0, 18, 35, 55, 75, 120], 
                         labels=['Pediatric (0-18)', 'Young Adult (18-35)', 'Middle-Aged (35-55)', 'Senior (55-75)', 'Geriatric (75+)'])
age_cost_distribution = var.groupby('Age_Group', observed=False)['Billing_Amount'].mean()
print("\nAverage Treatment Cost Across Age Demographics ($):")
print(age_cost_distribution)

# --- TASK D: DOCTOR CAPATIONAL EFFICIENCY LEADERBOARD ---
doctor_metrics = var.groupby('Doctor').agg({
    'Length_of_Stay_Days': 'mean',
    'Billing_Amount': ['mean', 'count'],
    'Clinical_Risk_Flag': 'mean'
})
# Clean multi-index headers down to simple flat titles
doctor_metrics.columns = ['Avg_Length_of_Stay', 'Avg_Billing_Amount', 'Total_Cases', 'Avg_Clinical_Risk_Flag']
doctor_metrics = doctor_metrics.sort_values(by='Total_Cases', ascending=False)

print("\nTop 5 Most Active Doctor Operational Profiles:")
print(doctor_metrics.head(5))

if not os.path.exists("output"):
    os.makedirs("output")

# --- PLOT 1: Box Plot (Treatment Costs across Disease Profiles) ---
plt.figure(figsize=(12, 6))
sns.boxplot(data=var, x='Medical_Condition', y='Billing_Amount', hue='Gender', palette='Set2', width=0.6)
plt.title('Financial Exposure Matrix: Treatment Billing Distribution by Disease Class', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Diagnosed Medical Condition')
plt.ylabel('Total Billing Amount ($)')
plt.tight_layout()
plt.savefig("output/chart1_treatment_cost_boxplot.png", dpi=300)
plt.close()

# --- PLOT 2: Histogram (Patient Age Distribution Density Curve) ---
plt.figure(figsize=(10, 6))
sns.histplot(data=var, x='Age', kde=True, color='darkslateblue', bins=20, edgecolor='black', alpha=0.7)
plt.axvline(var['Age'].mean(), color='red', linestyle='dashed', linewidth=2, label=f"Mean: {var['Age'].mean():.1f} Yrs")
plt.title('Clinical Demographics: Patient Age Influx Distribution Profile', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Patient Age (Years)')
plt.ylabel('Admissions Count Volume')
plt.legend()
plt.tight_layout()
plt.savefig("output/chart2_patient_age_histogram.png", dpi=300)
plt.close()

# --- PLOT 3: Pie Chart (Visual Simulation via Grouped Disease Share) ---
plt.figure(figsize=(8, 8))
plt.pie(condition_counts.values, labels=condition_counts.index, autopct='%1.1f%%', 
        startangle=140, colors=sns.color_palette('pastel')[0:len(condition_counts)],
        wedgeprops={'edgecolor': 'black', 'linewidth': 1})
plt.title('Epidemiology Matrix: Proportionate Share of Patient Conditions', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("output/chart3_disease_distribution_piechart.png", dpi=300)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(22, 16))

# Panel 1: Treatment Billing Distribution (Box Plot Layout)
sns.boxplot(data=var, x='Medical_Condition', y='Billing_Amount', palette='Set2', ax=axes[0, 0], width=0.5)
axes[0, 0].set_title('Treatment Billing Distribution by Disease Class', fontsize=13, fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Medical Condition Category')
axes[0, 0].set_ylabel('Total Billing ($)')

# Panel 2: Demographics Accumulation Profile (Histogram Layout)
sns.histplot(data=var, x='Age', kde=True, color='teal', bins=20, ax=axes[0, 1], edgecolor='black')
axes[0, 1].set_title('Global Patient Demographics Age Profile Influx', fontsize=13, fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Patient Age (Years)')

# Panel 3: Correlation Space Framework Matrix (Heatmap Map)
clinical_features = ['Age', 'Billing_Amount', 'Length_of_Stay_Days', 'Cost_Per_Day', 'Clinical_Risk_Flag']
correlation_space = var[clinical_features].corr()
sns.heatmap(correlation_space, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1, square=True, ax=axes[1, 0], cbar=False)
axes[1, 0].set_title('Operational Efficiency Matrix Covariance Space', fontsize=13, fontweight='bold', pad=15)

# Panel 4: Admission Type Intensity Distribution (Count Plot Grid)
sns.countplot(data=var, x='Admission_Type', hue='Test_Results', palette='viridis', edgecolor='black', ax=axes[1, 1])
axes[1, 1].set_title('Admission Urgency Profiling Split by Lab Test Severity', fontsize=13, fontweight='bold', pad=15)
axes[1, 1].set_xlabel('Urgency Tier Category')
axes[1, 1].set_ylabel('Admissions Count Volume')

plt.tight_layout()
plt.savefig("output/healthcare_executive_master_dashboard.png", dpi=300, bbox_inches='tight')
print(" -> [Saved Executive Master Dashboard] -> 'output/healthcare_executive_master_dashboard.png'")
plt.close()


print("1. THE CAPACITY BED-MAXIMIZATION TRAP:")
avg_los = var['Length_of_Stay_Days'].mean()
print(f"   - Global Systemic Average Length of Stay (LOS): {avg_los:.2f} Days per Patient")
print("   Corporate Narrative: Total billing curves exhibit flat, uniformly distributed variances across all disease categories.")
print("   This pattern strongly highlights that the underwriting model operates on fixed diagnostic group pricing codes.")
print("   Strategic Actionable: Financial growth depends entirely on accelerating safe discharge protocols and minimizing administrative lag rather than keeping patients bedded longer.")
print("2. PAYER MARGIN MANAGEMENT:")
lowest_insurance = insurance_margins['mean'].idxmin()
print(f"   - Payer provider generating lowest revenue per encounter: {lowest_insurance} (${insurance_margins.loc[lowest_insurance, 'mean']:.2f})")
print(f"   Strategic Verdict: Review contract parameters for '{lowest_insurance}' to ensure hospital operating cost margins remain protected against inflation adjustments.")