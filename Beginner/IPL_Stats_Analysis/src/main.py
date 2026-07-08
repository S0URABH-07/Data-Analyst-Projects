import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== STEP 1: READING IPL DATA ===")
df = pd.read_csv("data/ipl_players.csv")
print(df)


print("\n=== STEP 2: CALCULATING STRIKE RATE AND ECONOMY ===")
df['Strike_Rate'] = (df['Runs'] / df['Balls_Faced']) * 100

df['Economy'] = df['Runs_Conceded'] / df['Overs_Bowled']

df = df.fillna(0)
df = df.replace([np.inf, -np.inf], 0)
print(df[['Player', 'Runs', 'Strike_Rate', 'Wickets', 'Economy']])


print("\n=== STEP 3: FINDING ORANGE CAP & PURPLE CAP ===")
orange_index = df['Runs'].idxmax()
orange_cap_holder = df.loc[orange_index, 'Player']
max_runs = df.loc[orange_index, 'Runs']

purple_index = df['Wickets'].idxmax()
purple_cap_holder = df.loc[purple_index, 'Player']
max_wickets = df.loc[purple_index, 'Wickets']

print(f" Orange Cap Holder: {orange_cap_holder} ({max_runs} runs)")
print(f" Purple Cap Holder: {purple_cap_holder} ({max_wickets} wickets)")


print("\n=== STEP 4: FILTERING TOP BATSMEN ===")
top_batsmen = df[df['Runs'] > 500]
top_batsmen_sorted = top_batsmen.sort_values(by='Runs', ascending=False)
print(top_batsmen_sorted[['Player', 'Team', 'Runs', 'Strike_Rate']])


print("\n=== STEP 5: CREATING CHARTS ===")
plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1) # 1 row, 3 columns, position 1
plt.bar(top_batsmen_sorted['Player'], top_batsmen_sorted['Runs'], color='orange', edgecolor='black')
plt.title('Top Batsmen Runs (Orange Cap Race)')
plt.xlabel('Player')
plt.ylabel('Runs')
plt.xticks(rotation=90) 


bowlers_df = df[df['Wickets'] > 0].head(10)
plt.subplot(1, 3, 2) 
plt.pie(bowlers_df['Wickets'], labels=bowlers_df['Player'], autopct='%1.1f%%', startangle=90)
plt.title('Wicket Distribution')


plt.subplot(1, 3, 3) 
active_bowlers_economy = df[df['Economy'] > 0]['Economy']
plt.hist(active_bowlers_economy, bins=4, color='purple', edgecolor='black')
plt.title('Distribution of Bowling Economy')
plt.xlabel('Economy Rate')
plt.ylabel('Number of Bowlers')

plt.tight_layout()

if not os.path.exists("output"):
    os.makedirs("output")

plt.savefig("output/ipl_metrics_chart.png", dpi=300)
print("\nChart image successfully saved in Output Folder")

plt.show()