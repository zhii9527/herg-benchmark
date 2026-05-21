import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, joblib
import shap

# Ensure correct import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipelines.step04_model_training import scaffold_split

def generate_shap_summary_plot():
    print("🚀 Generating SHAP Feature Importance Plot for Weighted XGBoost...")

    # 1. Data preparation (using internal test set to explain model generalization)
    df = pd.read_parquet(config.SLIM_FEATURES_PARQUET)
    _, test_df = scaffold_split(df)

    meta_cols = ['std_smiles', 'Label', 'Source', 'pains_flag', 'consensus_flag', 'final_conf_flag', 'inchikey']
    X_test = test_df.drop(columns=meta_cols, errors='ignore')

    # 2. Load trained Weighted XGBoost model
    print("📦 Loading Assay-Weighted XGBoost model...")
    xgb_model = joblib.load(config.get_model_path("hERG_weighted_xgb_model.joblib"))

    # 3. Calculate SHAP values
    print("🧠 Calculating SHAP values (this might take a minute)...")
    # Use TreeExplainer, highly optimized for tree-based models
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # 4. Plot and save SHAP Summary Plot
    plt.figure(figsize=(10, 8))

    # Generate summary plot with colored points (top 20 features)
    shap.summary_plot(shap_values, X_test, max_display=20, show=False)

    # Get current axis and beautify
    ax = plt.gca()
    ax.set_title("SHAP Feature Importance (Assay-Weighted XGBoost)", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("SHAP value (Impact on model output: ← Safe | Toxic →)", fontsize=12)

    plt.tight_layout()
    plot_path = os.path.join(config.FIGURE_DIR, "shap_summary_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ SHAP explanation plot successfully saved to: {plot_path}")

if __name__ == "__main__":
    generate_shap_summary_plot()