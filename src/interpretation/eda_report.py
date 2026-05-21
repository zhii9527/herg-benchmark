import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure correct import path (pointing to src directory)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set global high-quality plotting style (publication standard)
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14})
sns.set_theme(style="whitegrid")

def run_herg_eda():
    print("\n🚀 [Phase 6.1] Starting hERG Exploratory Data Analysis (EDA) engine...")

    if not os.path.exists(config.SLIM_FEATURES_PARQUET):
        print("❌ Error: Cannot find the slim features table. Please run the data extraction phase first.")
        return

    df = pd.read_parquet(config.SLIM_FEATURES_PARQUET)

    # Create output directory
    eda_dir = os.path.join(config.FIGURE_DIR, "eda_plots")
    os.makedirs(eda_dir, exist_ok=True)

    # 1. MolLogP (Lipophilicity) Distribution Plot
    if 'MolLogP' in df.columns:
        print("   ▶ Plotting MolLogP distribution...")
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x='MolLogP', hue='Label', fill=True, palette=['#8EBAD9', '#E57F84'], alpha=0.5,
                    common_norm=False)
        plt.title("Chemical Logic: MolLogP Distribution by Toxicity")
        plt.xlabel("Lipophilicity (MolLogP)")
        plt.ylabel("Density")
        plt.text(df['MolLogP'].max() * 0.5, 0.1, "Toxic molecules shift\ntowards higher LogP", color='darkred',
                 weight='bold')
        plt.savefig(os.path.join(eda_dir, "EDA_1_MolLogP_KDE.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Molecular Weight (MolWt) Boxplot
    if 'MolWt' in df.columns:
        print("   ▶ Plotting MolWt boxplot...")
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='Label', y='MolWt', hue='Label', palette=['#8EBAD9', '#E57F84'], legend=False)
        plt.title("Molecular Weight Variance by Class")
        plt.xticks([0, 1], ['Safe (0)', 'Toxic (1)'])
        plt.savefig(os.path.join(eda_dir, "EDA_2_MolWt_Boxplot.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Top 12 Features Correlation Heatmap
    print("   ▶ Plotting feature correlation heatmap...")
    meta_cols = ['std_smiles', 'Label', 'Source', 'pains_flag', 'consensus_flag', 'final_conf_flag', 'inchikey']
    X = df.drop(columns=meta_cols, errors='ignore')

    top_feats = X.columns[:12]  # Select top 12 features
    corr_matrix = X[top_feats].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Inter-feature Correlation Map (Top Descriptors)")
    plt.savefig(os.path.join(eda_dir, "EDA_3_Correlation_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ EDA report generated successfully! Plots saved to: {eda_dir}")

if __name__ == "__main__":
    run_herg_eda()