# Task 2: Data Cleaning & Type Conversion

import pandas as pd

var = pd.read_csv("data/NetflixOriginals.csv")

# A. Convert a text date column into an actual datetime object
var['Premiere_Cleaned'] = pd.to_datetime(var['Premiere'], errors='coerce')
print(var['Premiere_Cleaned'])


# B. Extract parts of a date into separate columns
var['Year'] = var['Premiere_Cleaned'].dt.year
var['Month'] = var['Premiere_Cleaned'].dt.month_name()
var['Day_of_Week'] = var['Premiere_Cleaned'].dt.day_name()  
print(var['Year'] , var['Month'] , var['Day_of_Week'])

# C. Drop columns you don't need
var1 = var.drop(columns=['GenreLabels', 'Table'])
print(var1)

# D. Remove trailing or leading spaces from text columns
var['Status'] = var['Status'].str.strip()