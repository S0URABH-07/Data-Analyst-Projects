# Task 6: Sorting and Ranking
# Ordering your data to find top or bottom performers.

import numpy as np
import pandas as pd
var = pd.read_csv("data/NetflixOriginals.csv")

# A. Sort by a single column (Highest episodes first)
sorted_by_episodes = var.sort_values(by='EpisodesParsed', ascending=False)

# B. Sort by multiple columns (Sort by Year oldest-first, then by MinLength longest-first)
multi_sort = var.sort_values(by=['Year', 'MinLength'], ascending=[True, False])

# C. Find the absolute largest or smallest entries quickly without full sorting
print(var.nlargest(5, 'EpisodesParsed')) # Top 5 shows with most episodes
print(var.nsmallest(5, 'MinLength'))     # Bottom 5 shortest shows