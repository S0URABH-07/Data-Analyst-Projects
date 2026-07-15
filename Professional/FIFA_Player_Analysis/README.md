# Enterprise FIFA Player Analytics & Scouting Intelligence (Advanced Portfolio)

An advanced sports science data engineering and asset-valuation optimization platform evaluating professional athlete statistics, age-peaking patterns, and club payroll liabilities using Seaborn.

## Core Engineered Architecture
* **Talent Value Matrix:** Generates custom optimization markers checking Capital Efficiency (`value_per_wage_ratio`) and `growth_potential` windows.
* **Macro Market Tiering:** Structures asset performance layers by applying quantile cuts over global market values (`value_eur`) rather than relying on raw player IDs.
* **Multi-Asset Delivery Suite:** Automatically isolates and saves 4 standalone data science graphics, 1 high-resolution multi-axis presentation dashboard, and explicit feature covariance mappings.

## How to Run:
1. Download the raw CSV and place it inside your workspace directory at `data/fifa_players.csv`.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn`
3. Execute engine: `python main.py`