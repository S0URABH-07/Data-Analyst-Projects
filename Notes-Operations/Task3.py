# Task 3: Filtering and Conditional Selection
import pandas as pd
var = pd.read_csv("data/NetflixOriginals.csv").head(100)

# A. Simple filtering: Find all rows where the show is still "Renewed"
renewed_shows = var[var['Status'] == 'Renewed']
print(renewed_shows)

# B. Multiple conditions (AND / OR): Shows that are English AND have more than 3 seasons
# Note: Use & for AND, and | for OR
Hindi_long_shows = var[(var['Language'] == 'Hindi') & (var['SeasonsParsed'] > 3)]
print("Hindi Long Shows:- \n",Hindi_long_shows)

# C. Finding values within a specific range: Movies between 60 and 90 minutes
medium_movies = var[var['MinLength'].between(60, 90)]
print("Medium Time Movies:- \n",medium_movies)

# D. Filtering by a list of items (.isin): Get only Drama or Comedy entries
drama_comedy = var[var['Genre'].isin(['Drama', 'Comedy'])]
print("Drama Comedy:- \n",drama_comedy)