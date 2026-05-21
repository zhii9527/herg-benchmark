import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib
import json
import scipy.stats as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    precision_score, recall_score
)

# Ensure correct import path (pointing to src directory)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipelines.step04_model_training import scaffold_split
from pipelines.step05_model_evaluation import extract_features_for_new_data

# Set global high-quality plotting style
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.linewidth': 1.5})
sns.set_theme(style="whitegrid")


# ==========================================
# 1. Statistical Testing: True DeLong Test
# ==========================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def delong_roc_test(y_true, preds_A, preds_B):
    """Calculate the true DeLong P-value to compare AUC differences between two models."""
    y_true, preds_A, preds_B = np.array(y_true), np.array(preds_A), np.array(preds_B)
    pos_idx, neg_idx = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    m, n = len(pos_idx), len(neg_idx)

    txA, tyA = compute_midrank(preds_A[pos_idx]), compute_midrank(preds_A[neg_idx])
    txB, tyB = compute_midrank(preds_B[pos_idx]), compute_midrank(preds_B[neg_idx])

    aucA, aucB = roc_auc_score(y_true, preds_A), roc_auc_score(y_true, preds_B)

    vA10 = txA / n - (m + 1.0) / (2.0 * n)
    vA01 = 1.0 - (tyA / m - (n + 1.0) / (2.0 * m))
    vB10 = txB / n - (m + 1.0) / (2.0 * n)
    vB01 = 1.0 - (tyB / m - (n + 1.0) / (2.0 * m))

    S10, S01 = np.cov(vA10, vB10), np.cov(vA01, vB01)
    S = S10 / m + S01 / n

    if S[0, 0] + S[1, 1] - 2 * S[0, 1] == 0:
        return 1.0

    z = (aucA - aucB) / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return 2 * st.norm.sf(abs(z))


# ==========================================
# 2. Model Pipeline: Load or Train & Cache
# ==========================================
def load_or_train_models(X_train, y_train):
    """Load existing models, or train and save them if missing to avoid refitting every run."""
    models = {}
    print("\n📦 [1/4] Loading / Training Models...")

    # (A) Ours: Load the trained weighted model (must exist)
    models["Weighted XGB (Ours)"] = joblib.load(config.get_model_path("hERG_weighted_xgb_model.joblib"))

    # (B) Baselines: Check cache, train if not found
    baselines = {
        "Unweighted XGB": xgb.XGBClassifier(**config.XGB_PARAMS),
        "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced', n_jobs=-1,
                                                random_state=42),
        "MLP Baseline": make_pipeline(StandardScaler(),
                                      MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True,
                                                    random_state=42))
    }

    for name, model in baselines.items():
        cache_path = config.get_model_path(f"baseline_{name.replace(' ', '_')}.joblib")
        if os.path.exists(cache_path):
            models[name] = joblib.load(cache_path)
            print(f"   ▶ Loaded cached {name}")
        else:
            print(f"   ▶ Training {name} from scratch...")
            model.fit(X_train, y_train)
            joblib.dump(model, cache_path)
            models[name] = model

    # (C) SOTA: MapLight (if available)
    maplight_path = config.get_model_path("catboost_model.joblib")
    if os.path.exists(maplight_path):
        models["MapLight (SOTA)"] = joblib.load(maplight_path)
        print("   ▶ Loaded MapLight (SOTA)")

    return models


# ==========================================
# 3. Plotting Modules
# ==========================================
def plot_roc_pr_curves(model_probs, y_future, colors, line_styles):
    print("🎨 [3/4] Plotting ROC and PR Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    for name, prob_ext in model_probs.items():
        # ROC
        fpr, tpr, _ = roc_curve(y_future, prob_ext)
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if "Ours" in name or "SOTA" in name else 1.5
        ax1.plot(fpr, tpr, color=colors[name], linestyle=line_styles[name], lw=lw,
                 label=f'{name} (AUC = {roc_auc:.4f})')

        # PR
        precision, recall, _ = precision_recall_curve(y_future, prob_ext)
        pr_auc = average_precision_score(y_future, prob_ext)
        ax2.plot(recall, precision, color=colors[name], linestyle=line_styles[name], lw=lw,
                 label=f'{name} (AP = {pr_auc:.4f})')

    # Config Axes
    ax1.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle=':')
    ax1.set(xlim=[-0.01, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate',
            title='Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right", fontsize=10)

    baseline = y_future.mean()
    ax2.plot([0, 1], [baseline, baseline], color='black', lw=1.5, linestyle=':',
             label=f'Random Baseline ({baseline:.2f})')
    ax2.set(xlim=[-0.01, 1.0], ylim=[0.0, 1.05], xlabel='Recall', ylabel='Precision',
            title='Precision-Recall (PR) Curve')
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURE_DIR, "benchmark_roc_pr_curves.png"), dpi=300, bbox_inches='tight')


def plot_confusion_matrices(cm_ours, cm_rf):
    print("🎨 [4/4] Plotting Confusion Matrices...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_ours, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1, annot_kws={"size": 14})
    ax1.set_title('Weighted XGB (Ours) Confusion Matrix', fontsize=14, pad=15)

    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax2, annot_kws={"size": 14})
    ax2.set_title('Random Forest Confusion Matrix', fontsize=14, pad=15)

    for ax in [ax1, ax2]:
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xticklabels(['Safe (0)', 'Toxic (1)'])
        ax.set_yticklabels(['Safe (0)', 'Toxic (1)'])

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURE_DIR, "benchmark_confusion_matrices.png"), dpi=300)


# ==========================================
# 4. Main Execution Flow
# ==========================================
def run_comprehensive_benchmark():
    print("\n🚀 Starting Unified Comprehensive Benchmark...")
    os.makedirs(config.FIGURE_DIR, exist_ok=True)

    # --- Data Prep ---
    df = pd.read_parquet(config.SLIM_FEATURES_PARQUET)
    train_df, test_df = scaffold_split(df)
    meta_cols = ['std_smiles', 'Label', 'Source', 'pains_flag', 'consensus_flag', 'final_conf_flag', 'inchikey']

    X_train, y_train = train_df.drop(columns=meta_cols, errors='ignore'), train_df['Label']
    X_test, y_test = test_df.drop(columns=meta_cols, errors='ignore'), test_df['Label']

    future_csv = os.path.join(config.RAW_DATA_DIR, "ChEMBL36_Future_Validation.csv")
    df_future = pd.read_csv(future_csv)
    with open(config.get_model_path(f"selected_features_{config.TARGET_FEATURE_COUNT}.json"), 'r') as f:
        selected_features = json.load(f)
    X_future = extract_features_for_new_data(df_future['canonical_smiles'], selected_features)
    y_future = df_future['Label']

    # --- Load Models ---
    models = load_or_train_models(X_train, y_train)

    # --- Evaluation Loop ---
    print("\n📊 [2/4] Evaluating Models & Running Statistical Tests...")
    results = []
    model_probs_ext = {}
    threshold = 0.2

    # Get our baseline probabilities for DeLong comparison
    our_prob_ext = models["Weighted XGB (Ours)"].predict_proba(X_future)[:, 1]

    for name, model in models.items():
        prob_int = model.predict_proba(X_test)[:, 1]
        prob_ext = model.predict_proba(X_future)[:, 1]
        model_probs_ext[name] = prob_ext

        # Metrics
        y_pred_ext = (prob_ext >= threshold).astype(int)
        auc_int = roc_auc_score(y_test, prob_int)
        auc_ext = roc_auc_score(y_future, prob_ext)
        pr_auc_ext = average_precision_score(y_future, prob_ext)
        f1_ext = f1_score(y_future, y_pred_ext)
        mcc_ext = matthews_corrcoef(y_future, y_pred_ext)

        # DeLong vs Ours
        p_val = "-"
        if name != "Weighted XGB (Ours)":
            p_val = f"{delong_roc_test(y_future, our_prob_ext, prob_ext):.4f}"

        results.append({
            "Model": name,
            "Internal AUC": f"{auc_int:.4f}",
            "External AUC": f"{auc_ext:.4f}",
            "PR-AUC (Ext)": f"{pr_auc_ext:.4f}",
            "F1@0.2 (Ext)": f"{f1_ext:.4f}",
            "MCC (Ext)": f"{mcc_ext:.4f}",
            "DeLong p-val vs Ours": p_val
        })

    # --- Export Report ---
    report_df = pd.DataFrame(results)
    print("\n" + "🏆" * 35)
    print("      FINAL BENCHMARK: INTERNAL VS EXTERNAL PERFORMANCE")
    print("🏆" * 35)
    print(report_df.to_string(index=False))
    print("=" * 105)

    report_path = config.get_report_path("benchmark_comprehensive_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"✅ CSV Table exported to: {report_path}")

    # --- Plotting ---
    colors = {
        "Weighted XGB (Ours)": "#d62728", "MapLight (SOTA)": "#9467bd",
        "Unweighted XGB": "#1f77b4", "Random Forest": "#2ca02c", "MLP Baseline": "#ff7f0e"
    }
    line_styles = {
        "Weighted XGB (Ours)": "-", "MapLight (SOTA)": "-",
        "Unweighted XGB": "--", "Random Forest": "-.", "MLP Baseline": ":"
    }

    plot_roc_pr_curves(model_probs_ext, y_future, colors, line_styles)

    # Extract predictions for confusion matrices
    y_pred_ours = (model_probs_ext["Weighted XGB (Ours)"] >= threshold).astype(int)
    y_pred_rf = (model_probs_ext["Random Forest"] >= threshold).astype(int)
    plot_confusion_matrices(confusion_matrix(y_future, y_pred_ours), confusion_matrix(y_future, y_pred_rf))

    print("\n✅ All benchmark evaluations and visualizations completed successfully!")


if __name__ == "__main__":
    run_comprehensive_benchmark()