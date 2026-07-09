# Task 1: Basic Structural Exploration
import pandas as pd

var = pd.read_csv("data/NetflixOriginals.csv")
# See the exact rows and columns count
print(var.shape)

# View Top or Bottom Rows
print(var.head(10))
print(var.tail())

# Get a summary of columns, non-null counts, and data types
print(var.info())

# Get quick statistical calculations (Mean, Min, Max) for numeric columns
print(var.describe())