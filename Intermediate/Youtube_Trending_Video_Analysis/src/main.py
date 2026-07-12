import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

var = pd.read_csv("data/youtube.csv")
print(var.isnull().sum())

# Ensure columns are treated as numerical values safely
var["views"] = pd.to_numeric(var["views"],errors="coerce")
var["likes"] = pd.to_numeric(var["likes"],errors="coerce")
var["comment_count"] = pd.to_numeric(var["comment_count"],errors="coerce")

var["trending_date"] = pd.to_datetime(var["trending_date"],errors="coerce")
var["publish_date"] =pd.to_datetime(var["publish_date"],errors="coerce")

# METRICS & ADVANCED BUSINESS QUESTIONS
var['Like_Ratio_%'] = np.where(var['views'] > 0, (var['likes'] / var['views']) * 100, 0)
var['Comment_Ratio_%'] = np.where(var['views'] > 0, (var['comment_count'] / var['views']) * 100, 0)

# Most Trending Categories (Volume of video appearances)
trending_categories = var['category_id'].value_counts()
print("\nTop 5 Most Trending Categories (By Category ID Volume):")
print(trending_categories.head(5))

# Most Viewed Videos leaderboard
top_viewed = var.sort_values(by='views', ascending=False).head(5)
print("\nTop 5 Most Viewed Videos:")
print(top_viewed[['title', 'channel_title', 'views']].to_string(index=False, max_colwidth=40))

# Category Performance Efficiency (Average Engagement Ratios)
category_performance = var.groupby('category_id').agg({
    'Like_Ratio_%': 'mean',
    'Comment_Ratio_%': 'mean',
    'views': 'mean'
})
print("\nSample Category Performance Ratios:")
print(category_performance.head(15))

# INTER-VARIABLE DATA CORRELATIONS (EDA)
numerical_cols = ['views', 'likes', 'comment_count', 'Like_Ratio_%', 'Comment_Ratio_%']
correlation_matrix = var[numerical_cols].corr()
print("\nPearson Correlation Matrix:")
print(correlation_matrix)

# Set a clean default Seaborn theme backdrop
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 6))

# Seaborn Scatter Plot (Views vs Likes)
plt.subplot(1, 2, 1)
sns.scatterplot(data=var, x='views', y='likes', hue='category_id', palette='viridis', alpha=0.6, legend=False)
plt.title('Audience Scaling: Views vs Likes', fontsize=12, fontweight='bold')
plt.xlabel('View Count')
plt.ylabel('Like Count')

# Seaborn Heatmap Matrix (Correlation Map)
plt.subplot(1, 2, 2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, square=True)
plt.title('Metric Correlation Matrix Space', fontsize=12, fontweight='bold')

plt.tight_layout()

if not os.path.exists("output"):
    os.makedirs("output")
plt.savefig("output/youtube_eda_dashboard.png", dpi=300)

# Seaborn Pairplot (Multi-Variable Grid Array) 
print("\n[Seaborn] Spawning Pairplot Grid matrix canvas...")
# Isolate the top 3 most common categories to keep the plot colors clean and readable
top_categories = var['category_id'].value_counts().head(3).index
pairplot_df = var[var['category_id'].isin(top_categories)].head(5)

# Create the pairwise multi-plot grid structure
pairplot_fig = sns.pairplot(data=pairplot_df[numerical_cols + ['category_id']], hue='category_id', palette='Set1', diag_kind='kde', height=2.5)
pairplot_fig.savefig("output/youtube_pairplot_matrix.png", dpi=300)

plt.show()