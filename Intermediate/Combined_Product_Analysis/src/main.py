import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read both files
var = pd.read_csv("data/Baby_Products.csv")
var1 = pd.read_csv("data/Air_Conditioners.csv")

# Add a 'Product_Type' column to each dataset before combining them
# This keeps track of where each row originally came from
var['Product_Type'] = "Baby Product"
var1['Product_Type'] = "Air Conditioner"

print(f"Baby Products Rows: {var.shape[0]}")
print(f"Air Conditioner Rows: {var1.shape[0]}")


df = pd.concat([var,var1], ignore_index=True)
print(f"Total Combined Rows: {df.shape[0]}")


df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors='coerce')


df['discount_price'] = df['discount_price'].astype(str).str.replace('₹', '').str.replace('$', '').str.replace(',', '')
df['discount_price'] = pd.to_numeric(df['discount_price'], errors='coerce')
df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace('$', '').str.replace(',', '')
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')

print("Missing values count before cleanup:")
print(df.isnull().sum())


df['ratings'] = df['ratings'].fillna(df['ratings'].mean())
df['no_of_ratings'] = df['no_of_ratings'].fillna(df['no_of_ratings'].mean())
df['discount_price'] = df['discount_price'].fillna(df['discount_price'].mean())
df['actual_price'] = df['actual_price'].fillna(df['actual_price'].mean())



# Average Rating Comparison between Baby Products vs ACs
type_averages = df.groupby('Product_Type')['ratings'].mean()
print("\nAverage Rating by Product Type:")
print(type_averages)

# Average Prices Comparison
type_prices = df.groupby('Product_Type')['discount_price'].mean()
print("\nAverage Discount Price by Product Type:")
print(type_prices)

# Find the absolute top-rated item from each category
baby_only = df[df['Product_Type'] == 'Baby Product']
top_baby_name = baby_only.loc[baby_only['ratings'].idxmax(), 'name']
top_baby_rating = baby_only['ratings'].max()

# Filter for AC products only
ac_only = df[df['Product_Type'] == 'Air Conditioner']
top_ac_name = ac_only.loc[ac_only['ratings'].idxmax(), 'name']
top_ac_rating = ac_only['ratings'].max()

print(f"\nTop Baby Item: {top_baby_name} ({top_baby_rating} Stars)")
print(f"\nTop AC Item: {top_ac_name} ({top_ac_rating} Stars)")

plt.figure(figsize=(14, 5))

# Bar Plot
plt.subplot(1, 2, 1)
plt.bar(type_prices.index, type_prices.values, color=['skyblue', 'lightcoral'], edgecolor='black', width=0.4)
plt.title('Average Discount Price Comparison')
plt.ylabel('Price ($)')

# Scatter Plot 
plt.subplot(1, 2, 2)
plt.scatter(baby_only['ratings'], baby_only['discount_price'], color='blue', alpha=0.5, label='Baby Products')

plt.scatter(ac_only['ratings'], ac_only['discount_price'], color='red', alpha=0.5, label='Air Conditioners')
plt.title('Price vs Rating Distribution')
plt.xlabel('Rating Score')
plt.ylabel('Discount Price ($)')
plt.yscale('log') # Log scale because AC prices are much higher than baby items
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

if not os.path.exists("output"):
    os.makedirs("output")
    
plt.savefig("output/combined_product_comparison.png", dpi=300)

plt.show()