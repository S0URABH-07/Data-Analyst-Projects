# Corporate Fraud Detection & Transactional Credit Asset Intelligence (Kaggle Framework)

An enterprise-grade, production-ready modular risk operations pipeline evaluating merchant processing nodes, continuous dollar transaction limits, late-night spending distributions, and demographic vulnerability factors using Seaborn.

## Core Architectural Modules
* **Modular Clean Pipeline Structure:** Features dedicated `data_loader`, `cleaning`, `analysis`, and `visualization` components to ensure maintainable, high-grade portfolio deployment.
* **Temporal & Demographic Feature Engineering:** Derives critical features including `TransactionHour`, account `Customer_Age` maps, and transaction day cycles from complex raw source logs.
* **Multi-Asset Delivery Suite:** Automatically isolates and exports 4 standalone analytics graphics, 1 comprehensive unified 4-panel executive dashboard, and final feature correlation spaces.

## Key Strategic Risk Discoveries
1. **The High-Value Category Vector:** Fraud events are not randomly distributed; they present high clustering concentrations in specific merchant categories (e.g., `shopping_net`, `grocery_pos`). Fraud transactions in these sectors present significantly higher average transaction sizes (`amt`) compared to standard consumer spending.
2. **The Nocturnal Operational Window:** While normal credit use patterns collapse to near zero during early morning hours (1 AM to 5 AM), fraudulent automated script actions display a sharp density rise during this timeline, allowing risk detection groups to implement automated checkout holds.

## How to Execute:
1. Save your raw data file at path directory: `data/raw/credit_card_transactions.csv`.
2. Install standard dependencies layout: `pip install -r requirements.txt`
3. Execute production orchestration script: `python main.py`