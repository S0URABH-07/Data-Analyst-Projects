# Task 4: Aggregations and Grouping
# This is where you summarize data to find patterns.

import pandas as pd
var = pd.read_csv("data/NetflixOriginals.csv")

# A. Count how many times each unique value appears
var1 = var['Language'].value_counts()
print(var1)

# B. Group by one column and find the average of another
# Find the average minimum length for each unique Genre
print(var.groupby('Genre')['MinLength'].count())

# C. Group by and calculate multiple metrics at once using .agg()
# Find the total episodes and max seasons for each Language
summary = var.groupby('Language').agg({'EpisodesParsed': 'sum', 'SeasonsParsed': 'max'})
print("Summary:-\n",summary)