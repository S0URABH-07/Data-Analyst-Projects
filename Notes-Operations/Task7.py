# Task 7: Handling Missing Data (NaNs)
# Locating and treating blank holes in the dataset.

import numpy as np
import pandas as pd
var = pd.read_csv("data/NetflixOriginals.csv")

# A. Check which columns have missing values and count them
print(var.isna().sum())

# B. Fill missing numerical values with the column's average (mean)
var['SeasonsParsed'] = var['SeasonsParsed'].fillna(var['SeasonsParsed'].mean())

# C. Fill missing text values with a default string placeholder
var['Language'] = var['Language'].fillna('Unknown Language')

# D. Drop any row where vital data like 'Title' is completely missing
var1 = var.dropna(subset=['Title'])
print(var1)