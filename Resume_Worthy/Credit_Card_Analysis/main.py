import os
from src.data_loader import load_transaction_data
from src.cleaning import clean_transaction_matrix
from src.analysis import perform_fraud_quantifications
from src.visualization import generate_analytical_plots

def run_pipeline():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    # 1. Modular Data Loading
    raw_var = load_transaction_data("data/raw/credit_card_transactions.csv")
    
    # 2. Pipeline Execution Data Hygiene Cleanse
    cleaned_var = clean_transaction_matrix(raw_var)
    
    # 3. Business Analytics & Feature Covariances Summary
    correlation_space = perform_fraud_quantifications(cleaned_var)
    
    # 4. Generate Reporting Asset Suite
    generate_analytical_plots(cleaned_var, correlation_space)
    
    # 5. Export cleaned dataset file to the processed directory folder
    cleaned_var.to_csv("data/processed/cleaned_transactions.csv", index=False)
if __name__ == "__main__":
    run_pipeline()