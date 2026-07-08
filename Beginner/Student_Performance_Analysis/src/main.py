import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== STEP 1: LOADING THE DATA ===")
df = pd.read_csv("data/students.csv")
print(df)


print("\n=== STEP 2: CHECKING AND FIXING MISSING VALUES ===")
print("Missing values count:")
print(df.isnull().sum())

df = df.fillna(0)
print("\nData after filling missing values with 0:")
print(df)


print("\n=== STEP 3: CALCULATING TOTALS & AVERAGES ===")
subject_columns = ['Maths', 'Physics', 'Chemistry']

df['Total'] = df[subject_columns].sum(axis=1)

df['Percentage'] = df[subject_columns].mean(axis=1)

print(df)


print("\n=== STEP 4: FINDING THE HIGHEST SCORER ===")
highest_percent = df['Percentage'].max()

topper_index = df['Percentage'].idxmax()

topper_name = df.loc[topper_index, 'Name']

print(f"Topper Student: {topper_name}")
print(f"Highest Score: {highest_percent}%")


print("\n=== STEP 5: SUBJECT-WISE CLASS AVERAGE ===")
subject_averages = df[subject_columns].mean()
print(subject_averages)


print("\n=== STEP 6: CREATING CHARTS ===")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) 
plt.bar(df['Name'], df['Percentage'], color='skyblue')
plt.title('Student Percentages')
plt.xlabel('Names')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100) 

plt.subplot(1, 2, 2)
plt.bar(subject_averages.index, subject_averages.values, color='salmon')
plt.title('Subject Averages')
plt.xlabel('Subjects')
plt.ylabel('Marks')
plt.ylim(0, 100)

plt.show()