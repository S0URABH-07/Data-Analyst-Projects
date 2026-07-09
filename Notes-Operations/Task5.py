# Task 5: Using NumPy for Conditional Logic
# NumPy is incredibly fast for adding new flags or columns based on conditions.

import numpy as np
import pandas as pd
var = pd.read_csv("data/NetflixOriginals.csv")

# A. Create a binary flag (1 or 0) using np.where
# The np.where() function requires three arguments, formatted like this:
# np.where(condition , value_if_True , Value_if_False)
var['Is_Active_Flag'] = np.where(var['Active'] == 1, 1, 0)

# B. Create multiple categories based on conditions using np.select
# Categorize shows based on length
conditions = [
    var['MinLength'] < 30,
    (var['MinLength'] >= 30) & (var['MinLength'] <= 60),
    var['MinLength'] > 60
]
categories = ['Short Form', 'Medium Form', 'Feature Length']

var['Length_Category'] = np.select(conditions, categories, default='Unknown')
# The np.select() function requires three main arguments, passed like this:
# np.select(condition List, Choice List, default Value)