import pandas as pd
import numpy as np
import os
import sys
import random
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

# Ensure correct import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ==========================================
# True Randomized Scaffold Split Function
# ==========================================
def randomized_scaffold_split(df, train_frac=0.8, random_state=42):
    """
    Groups molecules by Murcko scaffold, randomizes the order of the scaffolds
    based on the seed, and splits them to ensure rigorous variance estimation.
    """
    random.seed(random_state)

    # 1. Group indices by scaffold
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        # Handle cases where SMILES might be invalid or NaN
        smi = str(row['std_smiles'])
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(smi)
            scaffolds[scaffold].append(idx)
        except:
            scaffolds['Generic_Invalid_Scaffold'].append(idx)

    # 2. Shuffle the scaffold groups randomly
    scaffold_sets = list(scaffolds.values())
    random.shuffle(scaffold_sets)

    # 3. Assign to Train and Test sets
    train_idx, test_idx = [], []
    train_cutoff = int(train_frac * len(df))

    for scaffold_group in scaffold_sets:
        if len(train_idx) + len(scaffold_group) <= train_cutoff:
            train_idx.extend(scaffold_group)
        else:
            test_idx.extend(scaffold_group)

    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def run_si_generation():
    print("🚀 Starting 5-Fold Randomized Scaffold Cross-Validation for SI...\n")

    # 1. Load the slimmed feature table
    if not os.path.exists(config.SLIM_FEATURES_PARQUET):
        print("❌ Error: Slim features table not found. Please ensure data is generated.")
        return

    df = pd.read_parquet(config.SLIM_FEATURES_PARQUET).reset_index(drop=True)

    # Extract real dimensionality data for Table S2
    meta_cols = ['std_smiles', 'Label', 'Source', 'pains_flag', 'consensus_flag', 'final_conf_flag', 'inchikey']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    morgan_count = sum(1 for c in feature_cols if "Morgan" in c)
    phys_count = sum(1 for c in feature_cols if "Morgan" not in c and "MACCS" not in c)
    maccs_count = sum(1 for c in feature_cols if "MACCS" in c)

    print("=" * 50)
    print("📊 [Table S2] Feature Dimensionality Output")
    print(f"   ▶ Selected Morgan Dimensions: {morgan_count}")
    print(f"   ▶ Selected RDKit/PhysChem Dimensions: {phys_count + maccs_count}")
    print("=" * 50 + "\n")

    # 2. Simulate 5-fold scaffold cross-validation
    print("⏳ Running 5 independent random scaffold splits to calculate true variance...")

    folds_data = []
    threshold = 0.2

    for fold in range(1, 6):
        # Now using our custom TRUE randomized scaffold split
        train_df, test_df = randomized_scaffold_split(df, train_frac=0.8, random_state=42 + fold)

        X_train = train_df.drop(columns=meta_cols, errors='ignore')
        y_train = train_df['Label']
        train_weights = train_df['final_conf_flag'].map({'high_confidence': 1.0, 'low_confidence': 0.5}).values

        X_test = test_df.drop(columns=meta_cols, errors='ignore')
        y_test = test_df['Label']

        # A. Train Weighted XGBoost (Ours)
        model_w_xgb = xgb.XGBClassifier(**config.XGB_PARAMS)
        model_w_xgb.fit(X_train, y_train, sample_weight=train_weights)
        prob_w_xgb = model_w_xgb.predict_proba(X_test)[:, 1]
        auc_w_xgb = roc_auc_score(y_test, prob_w_xgb)
        mcc_w_xgb = matthews_corrcoef(y_test, (prob_w_xgb >= threshold).astype(int))

        # B. Train Unweighted XGBoost
        model_xgb = xgb.XGBClassifier(**config.XGB_PARAMS)
        model_xgb.fit(X_train, y_train)
        prob_xgb = model_xgb.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, prob_xgb)
        mcc_xgb = matthews_corrcoef(y_test, (prob_xgb >= threshold).astype(int))

        # C. Train Random Forest Baseline
        model_rf = RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced', random_state=42,
                                          n_jobs=-1)
        model_rf.fit(X_train, y_train)
        prob_rf = model_rf.predict_proba(X_test)[:, 1]
        auc_rf = roc_auc_score(y_test, prob_rf)
        mcc_rf = matthews_corrcoef(y_test, (prob_rf >= threshold).astype(int))

        folds_data.append({
            "Fold": f"Fold {fold}",
            "W_XGB_AUC": auc_w_xgb, "W_XGB_MCC": mcc_w_xgb,
            "XGB_AUC": auc_xgb, "XGB_MCC": mcc_xgb,
            "RF_AUC": auc_rf, "RF_MCC": mcc_rf
        })
        print(f"   ✓ {fold}/5 folds completed.")

    # 3. Statistical Aggregation and Export
    df_res = pd.DataFrame(folds_data)

    print("\n" + "=" * 70)
    print("🏆 [Table S3] 5-Fold Cross-Validation Performance Report")
    print("=" * 70)
    for _, r in df_res.iterrows():
        print(f"{r['Fold']}:")
        print(f"  - Weighted XGB (Ours) : AUC = {r['W_XGB_AUC']:.4f} | MCC = {r['W_XGB_MCC']:.4f}")
        print(f"  - Unweighted XGB     : AUC = {r['XGB_AUC']:.4f} | MCC = {r['XGB_MCC']:.4f}")
        print(f"  - Random Forest      : AUC = {r['RF_AUC']:.4f} | MCC = {r['RF_MCC']:.4f}")

    print("-" * 70)
    print("Mean ± SD Summary:")

    summary_data = []
    models = [
        ("Weighted XGB (Ours)", "W_XGB_AUC", "W_XGB_MCC"),
        ("Unweighted XGB", "XGB_AUC", "XGB_MCC"),
        ("Random Forest", "RF_AUC", "RF_MCC")
    ]

    for name, auc_col, mcc_col in models:
        mean_auc, std_auc = df_res[auc_col].mean(), df_res[auc_col].std()
        mean_mcc, std_mcc = df_res[mcc_col].mean(), df_res[mcc_col].std()
        print(f"  - {name:<20}: AUC = {mean_auc:.4f} ± {std_auc:.4f} | MCC = {mean_mcc:.4f} ± {std_mcc:.4f}")

        summary_data.append({
            "Model": name,
            "AUC (Mean ± SD)": f"{mean_auc:.4f} ± {std_auc:.4f}",
            "MCC (Mean ± SD)": f"{mean_mcc:.4f} ± {std_mcc:.4f}"
        })
    print("=" * 70)

    # Export to CSV
    os.makedirs(config.FIGURE_DIR, exist_ok=True)
    summary_df = pd.DataFrame(summary_data)
    export_path = config.get_report_path("Table_S3_CV_Results.csv")
    summary_df.to_csv(export_path, index=False)
    print(f"\n✅ Table S3 successfully exported to: {export_path}")


if __name__ == "__main__":
    run_si_generation()