import pandas as pd
import numpy as np
import os
import sys
import xgboost as xgb
import joblib
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, matthews_corrcoef
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

# Ensure parent directory is accessible for global configuration imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_scaffold(smiles):
    """Generates the Bemis-Murcko scaffold for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.GetScaffoldForMol(mol) if mol else ""
    except:
        return ""


def scaffold_split(df, train_frac=0.8, random_state=42):
    """Performs Bemis-Murcko scaffold-based splitting to maintain strict structural isolation."""
    print("   🧬 Generating Bemis-Murcko scaffolds for structural data partitioning...")
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold = generate_scaffold(row['std_smiles'])
        scaffolds[scaffold].append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    train_indices, test_indices = [], []
    n_train_target = int(len(df) * train_frac)

    for scaffold_set in scaffold_sets:
        if len(train_indices) + len(scaffold_set) <= n_train_target:
            train_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)
    return df.iloc[train_indices], df.iloc[test_indices]


def calculate_sample_weights(df):
    """
    Applies a tiered sample weighting logic optimized for multi-source TDC data integration.
    """
    print("   ⚖️ Applying TDC-specific layered weights...")

    def get_base_weight(source_str):
        source_str = str(source_str).upper()
        # 1. Assign highest confidence weight to rigorous literature-extracted ChEMBL data
        if 'CHEMBL' in source_str: return 1.0
        # 2. Assign high weight to curated Karim benchmark dataset
        if 'KARIM' in source_str: return 0.8
        # 3. Assign lower weight to massive but high-noise hERGCentral HTS data
        if 'CENTRAL' in source_str: return 0.5
        # 4. Assign baseline weight to other standard TDC raw sources
        if 'TDC' in source_str: return 0.6
        return 0.5

    # Compute data provenance weights
    s_weights = df['Source'].apply(get_base_weight)

    # Compute data curation confidence weights (high=1.0, low=0.3 derived from Phase 1 data audit)
    c_weights = df['final_conf_flag'].map(config.CONFIDENCE_WEIGHTS).fillna(0.3)

    # Composite weight calculation = provenance_weight * curation_confidence_weight
    final_weights = s_weights * c_weights

    # Log weight matrix boundaries for pipeline validation
    print(
        f"   ▶ Weight Summary: Max={final_weights.max():.2f}, Min={final_weights.min():.2f}, Mean={final_weights.mean():.3f}")
    return final_weights


def run_training():
    """Pipeline Phase 4: Scaffold Partitioning and Weighted XGBoost Architecture Optimization."""
    print(f"\n🚀 [Phase 4] Starting Layered Weighted Model Training...")

    if not os.path.exists(config.SLIM_FEATURES_PARQUET):
        print(f"❌ Error: Slim feature table {config.SLIM_FEATURES_PARQUET} not found. Please run 03_feature_selection.py first.")
        return

    # 1. Load dataset
    df = pd.read_parquet(config.SLIM_FEATURES_PARQUET)

    # 2. Enforce scaffold-based dataset partitioning
    train_df, test_df = scaffold_split(df)
    print(f"   ▶ Train Samples: {len(train_df)}, Test Samples: {len(test_df)}")

    # 3. Segregate feature matrices, target vectors, and metadata
    meta_cols = ['std_smiles', 'Label', 'Source', 'pains_flag', 'consensus_flag', 'final_conf_flag', 'inchikey']
    X_train = train_df.drop(columns=meta_cols, errors='ignore')
    y_train = train_df['Label']
    X_test = test_df.drop(columns=meta_cols, errors='ignore')
    y_test = test_df['Label']

    # Compute composite sample weights strictly on the training partition
    train_weights = calculate_sample_weights(train_df)

    # 4. Fit Regularized XGBoost Classifier
    print(f"\n⚡ Training XGBoost on {config.XGB_PARAMS.get('device', 'cpu')} (Applying Strong Regularization)...")
    model = xgb.XGBClassifier(**config.XGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=train_weights)

    # 5. Validation metrics evaluation at a fixed empirical threshold of 0.20
    eval_threshold = 0.20
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= eval_threshold).astype(int)

    auc_roc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("\n" + "✅" * 20)
    print("    SCAFFOLD VALIDATION PERFORMANCE")
    print("✅" * 20)
    print(f"ROC-AUC:  {auc_roc:.4f}")
    print(f"PR-AUC:   {auc_pr:.4f}")
    print(f"F1-Score: {f1:.4f} (at empirical threshold {eval_threshold})")
    print(f"MCC:      {mcc:.4f} (at empirical threshold {eval_threshold})")
    print("=" * 40)

    # 6. Serialize optimized model artifacts and feature configurations
    model_path = config.get_model_path("hERG_weighted_xgb_model.joblib")
    feat_path = config.get_model_path(f"selected_features_{config.TARGET_FEATURE_COUNT}.json")

    joblib.dump(model, model_path)
    with open(feat_path, 'w') as f:
        json.dump(X_train.columns.tolist(), f)

    print(f"\n✅ Phase 4 Complete! Model & Feature mapping saved to: {config.MODEL_DIR}")


if __name__ == "__main__":
    import time

    start = time.time()
    run_training()
    print(f"\n🏁 Total execution time: {(time.time() - start) / 60:.2f} minutes")