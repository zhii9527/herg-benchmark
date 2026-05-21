import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

# Ensure correct import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipelines.step05_model_evaluation import extract_features_for_new_data

def generate_tsne_space_plot():
    print("🚀 Generating Chemical Space Projection (t-SNE + KDE)...")

    # 1. Load Internal Dataset
    print("📦 Loading Internal Dataset...")
    df_internal = pd.read_parquet(config.SLIM_FEATURES_PARQUET)
    meta_cols = ['std_smiles', 'Label', 'Source', 'pains_flag', 'consensus_flag', 'final_conf_flag', 'inchikey']
    X_internal = df_internal.drop(columns=meta_cols, errors='ignore')

    # 2. Load External OOD Dataset
    print("📦 Loading External OOD Dataset...")
    future_csv = os.path.join(config.RAW_DATA_DIR, "ChEMBL36_Future_Validation.csv")
    df_future = pd.read_csv(future_csv)

    with open(config.get_model_path(f"selected_features_{config.TARGET_FEATURE_COUNT}.json"), 'r') as f:
        selected_features = json.load(f)
    X_external = extract_features_for_new_data(df_future['canonical_smiles'], selected_features)

    # 3. Balanced downsampling (for t-SNE speed and to prevent plot overcrowding)
    sample_size = min(len(X_external), 5000)
    print(f"⚖️ Sampling {sample_size} compounds from each set for clear visualization...")
    X_internal_sampled = X_internal.sample(n=sample_size, random_state=42)
    X_external_sampled = X_external.sample(n=sample_size, random_state=42) if len(X_external) > sample_size else X_external

    df_pca_internal = pd.DataFrame(X_internal_sampled)
    df_pca_internal['Dataset'] = 'Internal Training Space'

    df_pca_external = pd.DataFrame(X_external_sampled)
    df_pca_external['Dataset'] = 'External OOD Space (ChEMBL 22-24)'

    df_combined = pd.concat([df_pca_internal, df_pca_external], axis=0).reset_index(drop=True)
    X_combined = df_combined.drop(columns=['Dataset'])
    y_dataset = df_combined['Dataset']

    # 4. Standardization -> PCA(50) -> t-SNE(2) (Industry standard acceleration pipeline)
    print("🧠 Performing Standard Scaler -> PCA(50) -> t-SNE(2)... This may take 1-2 minutes...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    pca_50 = PCA(n_components=min(50, X_scaled.shape[1]), random_state=42)
    X_pca = pca_50.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_pca)

    df_plot = pd.DataFrame({
        't-SNE 1': X_tsne[:, 0],
        't-SNE 2': X_tsne[:, 1],
        'Dataset': y_dataset
    })

    # 5. Plot chemical space distribution with density contours
    print("🎨 Plotting high-quality KDE + Scatter plot...")
    plt.figure(figsize=(11, 8.5))

    palette = {'Internal Training Space': '#90A4AE', 'External OOD Space (ChEMBL 22-24)': '#E53935'}

    # Plot KDE density contours (showing shift in distribution center)
    sns.kdeplot(
        data=df_plot, x='t-SNE 1', y='t-SNE 2', hue='Dataset',
        palette=palette, alpha=0.7, levels=5, linewidths=2, fill=False
    )

    # Plot underlying scatter (showing individual scaffold islands)
    sns.scatterplot(
        data=df_plot.sort_values('Dataset', ascending=False),
        x='t-SNE 1', y='t-SNE 2', hue='Dataset',
        palette=palette, alpha=0.4, s=15, edgecolor=None, legend=False
    )

    plt.title("Chemical Space Distribution (t-SNE Projection & Density)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=12, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=12, fontweight='bold')

    # Manually add legend for clarity
    patch_int = mpatches.Patch(color=palette['Internal Training Space'], label='Internal Training Space')
    patch_ext = mpatches.Patch(color=palette['External OOD Space (ChEMBL 22-24)'], label='External OOD Space (ChEMBL 22-24)')
    plt.legend(handles=[patch_int, patch_ext], title='Data Source', title_fontsize='13', fontsize='11', loc='best', frameon=True, shadow=True)

    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(config.FIGURE_DIR, "chemical_space_tsne.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ t-SNE chemical space plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    generate_tsne_space_plot()