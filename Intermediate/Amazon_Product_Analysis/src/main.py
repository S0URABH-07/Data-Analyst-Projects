import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

var = pd.read_csv("data/Baby_Products.csv")
print(var.columns.to_list())

print(var.isnull().sum())

# Make column numeri
var["ratings"] = pd.to_numeric(var["ratings"],errors="coerce")
var["no_of_ratings"] = pd.to_numeric(var["no_of_ratings"],errors='coerce')

# Replace string in numeric column
var["discount_price"] = var["discount_price"].astype(str).str.replace("₹","").str.replace("$","").str.replace(",","")
var["discount_price"] = pd.to_numeric(var["discount_price"],errors="coerce")

var["actual_price"] = var["actual_price"].astype(str).str.replace("₹","").str.replace("$","").str.replace(",","")
var["actual_price"] = pd.to_numeric(var["actual_price"],errors="coerce")


# Fill Empty value
var["ratings"] = var["ratings"].fillna(var["ratings"].mean())
var["no_of_ratings"] = var["no_of_ratings"].fillna(var["no_of_ratings"].mean())
var["discount_price"] = var["discount_price"].fillna(var["discount_price"].min())
var["actual_price"] = var["actual_price"].fillna(var["actual_price"].max())

print(var.isnull().sum())

# Top 5 Most rated Products
highest_rated = var.sort_values(by=["ratings","no_of_ratings"] , ascending=[False,False]).head(5)
print(highest_rated[["name","ratings","no_of_ratings"]].to_string(index=False , max_colwidth=40))

# Find discount price
var["discount_amount"] = var["actual_price"] - var["discount_price"]

# Find Best Discount Deal
best_discount_idx = var["discount_amount"].idxmax()
best_discount_product_name = var.loc[best_discount_idx,"name"]
max_saved_amount = var.loc[best_discount_idx , "discount_amount"]
print(f"Best Deal :- Save {max_saved_amount:,.2f} on {best_discount_product_name}")


plt.figure(figsize=(15,10))

# Scatter Plot
plt.subplot(1,2,1)
plt.scatter(var["ratings"] , var["no_of_ratings"] , color="deepskyblue" , edgecolor="black")
plt.title('Scatter Plot: Ratings vs Review Volume')
plt.xlabel('Rating Score (Out of 5)')
plt.ylabel('Number of Ratings (Log Scale)')
plt.yscale('log') # Scale view vertically because engagement numbers span dynamically
plt.grid(True, linestyle='--', alpha=0.5)

#  Correlation Heatmap
plt.subplot(1, 2, 2)
numeric_columns = ['ratings', 'no_of_ratings', 'discount_price', 'actual_price']
correlation_matrix = var[numeric_columns].corr()

plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.title('E-commerce Metric Correlation')
plt.colorbar()

# Map labels cleanly down spatial edges
ticks = np.arange(len(correlation_matrix.columns))
plt.xticks(ticks, correlation_matrix.columns, rotation=15)
plt.yticks(ticks, correlation_matrix.columns)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        val = correlation_matrix.iloc[i, j]
        plt.text(i, j, f"{val:.2f}", ha='center', va='center', color='black', fontweight='bold')

plt.tight_layout()

if not os.path.exists("output"):
    os.makedirs("output")
    
plt.savefig("output/baby_products_dashboard.png", dpi=300)

plt.show()