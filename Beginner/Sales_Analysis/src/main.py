import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Read Sales Data")
var = pd.read_csv("data/sales_data.csv")
# print(var)

# print columns name in list
print(var.columns.tolist())

# Total revenue
total_revenue = var["Revenue"].sum()
print(f"Total revenue :- {total_revenue:,.2f}")

# Monthly revenue
var["Year_Month"] = var["Year"].astype(str) + "-" + var["Month"]
monthly_revenue = var.groupby("Year_Month")["Revenue"].sum()
print(f"Monthly revenue is {monthly_revenue.head(5)}")


# product base total revenue
product_revenue = var.groupby("Product")["Revenue"].sum()

# bast product
best_product = product_revenue.idxmax()
best_product_sales = product_revenue.max()
print(f"Best performing Product:- {best_product} {best_product_sales}")

# worst product
worst_product = product_revenue.idxmin()
worst_product_sales = product_revenue.min()
print(f"Worst performing Product:- {worst_product} {worst_product_sales}")

# Country wise sales total
country_sales = var.groupby("Country")["Revenue"].sum()
print("Revenue Breakdown by Country:- ",country_sales)
plt.figure(figsize=(15, 10))
# Line chart
plt.subplot(1,3,1)
monthly_revenue_sorted = monthly_revenue.sort_index().head(20)
plt.plot(monthly_revenue_sorted.index , monthly_revenue_sorted.values , marker="o" , color="#2ca02c" , linewidth=2)
plt.title("Monthly Revenue Tracking")
plt.xlabel("Timeline Months")
plt.ylabel("Revenue")
plt.xticks(rotation=90,fontsize=8)
plt.grid(True,linestyle="--" , alpha =0.5)

# Bar chart
plt.subplot(1,3,2)
top_5_products = product_revenue.sort_values(ascending=False).head(5)
plt.bar(top_5_products.index, top_5_products.values, color='#1f77b4', edgecolor='black')
plt.title('Top 5 Products by Revenue')
plt.xlabel('Product Description')
plt.ylabel('Revenue')
plt.xticks(rotation=45, ha='right', fontsize=9)

# Pie Chart
plt.subplot(1, 3, 3)
plt.pie(country_sales.values, labels=country_sales.index, autopct='%1.1f%%', startangle=140)
plt.title("Country-wise Contribution Split")

plt.tight_layout()

if not os.path.exists("output"):
    os.makedirs("output")
    
plt.savefig("output/sales_dashboard_summary.png", dpi=300)

plt.show()