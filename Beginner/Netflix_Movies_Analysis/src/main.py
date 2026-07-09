import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os

print("Read AND Clean the data")
var = pd.read_csv("data/NetflixOriginals.csv").head(50)
# print(var)

print("Availavle Column in dataset")
print(var.columns.tolist())

print("Extract release year")
var["Premiere_Clean"] = pd.to_datetime(var["Premiere"] , errors="coerce")
var["Release_Year"] = var["Premiere_Clean"].dt.year

# Which Genre is most popular
genre_count = var["Genre"].value_counts()
popular_genre = genre_count.idxmax()
highest_genre_count = genre_count.max()
print(f"popular genre {popular_genre} {highest_genre_count} titles")

# Which year relese most movies
year_count = var["Release_Year"].dropna().value_counts()
top_year = int(year_count.idxmax())
most_movies_count = year_count.max()
print(f"most releases year {top_year} {most_movies_count} titles")

# Average movie duration 
average_duration = var["MinLength"].mean()
print(f"Average movie duration is {average_duration} minutes")

# Top countries Uses Language
top_languages = var["Language"].value_counts()
print(f"Top 5 languages:- {top_languages}")

print("Create Visualization\n")

# bar plot 
plt.subplot(1,2,1)
top_7_genre = genre_count.head(7)
plt.bar(top_7_genre.index , top_7_genre.values , color="crimson" , edgecolor="black")
plt.title("Top 7 Popular Genres")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Histogram
plt.subplot(1,2,2)
plt.hist(var["MinLength"].dropna(),bins=15,color="darkgray",edgecolor="black")
plt.axvline(average_duration,color="red",linestyle="dashed",linewidth=2,label=f"Avg: {average_duration:.1f}m")
plt.title("Distribution of Movie Duration")
plt.xlabel("Duration : (minutes)")
plt.ylabel("Number of Titles")
plt.legend()

plt.tight_layout()

if not os.path.exists("output"):
    os.makedirs("output")

plt.savefig("output/netflix_originals_plot.png",dpi=300)
print("Successfully Saved Your Image")
plt.show()