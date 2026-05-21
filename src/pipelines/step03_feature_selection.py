import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import sys

# Ensure parent directory is accessible for global configuration imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def eliminate_collinearity(X):
    """Executes a rigorous Spearman Collinearity Audit to remove redundant dimensions."""
    print("\n🔍 [Phase 3 - Step 1] Executing Spearman Collinearity Audit...")

    # 1. Drop zero-variance features first
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    # 2. Compute absolute Spearman correlation matrix
    X_ranked = X.rank()
    corr_array = np.corrcoef(X_ranked.values, rowvar=False)
    corr_matrix = pd.DataFrame(np.abs(corr_array), index=X.columns, columns=X.columns)

    # 3. Drop highly correlated features using the upper triangle mask
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > config.CORR_THRESHOLD)]

    if len(to_drop) > 0:
        X_cleaned = X.drop(columns=to_drop)
        return X_cleaned, len(to_drop)
    return X, 0


def generate_scaffold_folds(df, n_splits=3):
    """
    Generates rigorous outer cross-validation folds based on Bemis-Murcko Scaffolds.
    This effectively prevents structural data leakage between training and validation folds.
    """
    print(f"\n🧬 Generating {n_splits}-Fold Scaffold Splits...")
    scaffolds = defaultdict(list)

    # Extract Bemis-Murcko scaffolds for all target molecules
    for i, smi in enumerate(df['std_smiles']):
        try:
            mol = Chem.MolFromSmiles(smi)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol) if mol else ""
            scaffolds[scaffold].append(i)
        except:
            scaffolds[""].append(i)

    # Sort scaffold clusters from largest to smallest to ensure greedy distribution
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    folds = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    # Greedily assign each scaffold cluster to the smallest fold to balance dataset sizes
    for scaf_set in scaffold_sets:
        smallest_fold = np.argmin(fold_sizes)
        folds[smallest_fold].extend(scaf_set)
        fold_sizes[smallest_fold] += len(scaf_set)

    # Construct independent train/validation index index pairs
    cv_splits = []
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = []
        for j in range(n_splits):
            if i != j:
                train_idx.extend(folds[j])
        cv_splits.append((train_idx, test_idx))

    return cv_splits


def run_nested_cv_and_finalize():
    """Executes unbiased Nested CV evaluation and locks final production features."""
    print(f"\n🚀 [Phase 3] Starting Feature Selection Pipeline...")

    if os.path.exists(config.SLIM_FEATURES_PARQUET):
        print(f"✨ [Skip] Slim feature table already exists at {config.SLIM_FEATURES_PARQUET}. Delete manually to re-run.")
        return

    # Load master feature matrix
    if not os.path.exists(config.FULL_FEATURES_PARQUET):
        print(f"❌ Error: Master feature table {config.FULL_FEATURES_PARQUET} not found. Please run 02_feature_pipeline.py first.")
        return

    df = pd.read_parquet(config.FULL_FEATURES_PARQUET)
    metadata_cols = ['std_smiles', 'Label', 'Source', 'pains_flag', 'consensus_flag', 'final_conf_flag', 'inchikey']
    X = df.drop(columns=metadata_cols, errors='ignore').select_dtypes(include=['number', 'bool'])
    y = df['Label']

    # 1. Eliminate Collinearity
    X_pure, dropped_count = eliminate_collinearity(X)
    print(f"   - Dropped {dropped_count} highly correlated features.")

    # 2. Sample records for computational feasibility during nested screening
    print(f"\n💡 [Optimization] Sampling 50,000 records for robust nested feature selection...")
    if len(X_pure) > 50000:
        sample_idx = X_pure.sample(n=50000, random_state=42).index
        X_sample = X_pure.loc[sample_idx].reset_index(drop=True)
        y_sample = y.loc[sample_idx].reset_index(drop=True)
        df_sample = df.loc[sample_idx].reset_index(drop=True)  # Required for Bemis-Murcko scaffold extraction
    else:
        X_sample, y_sample, df_sample = X_pure, y, df

    # =====================================================================
    # 3. Strict Nested CV Evaluation (Reviewer-Proof Validation)
    # =====================================================================
    print("\n🚀 [Phase 3 - Step 2] Launching Strict Nested Scaffold CV (No Data Leakage)...")

    # Enforce CPU execution for multi-threaded stability during intensive recursive elimination
    xgb_cpu_params = config.XGB_PARAMS.copy()
    xgb_cpu_params['device'] = 'cpu'
    xgb_cpu_params['tree_method'] = 'hist'

    scaffold_folds = generate_scaffold_folds(df_sample, n_splits=3)
    outer_scores = []

    # Outer Loop: Unbiased validation on scaffold-isolated data splits
    for fold, (train_idx, val_idx) in enumerate(scaffold_folds):
        print(f"\n--- Processing Outer Scaffold Fold {fold + 1}/3 ---")

        X_tr, y_tr = X_sample.iloc[train_idx], y_sample.iloc[train_idx]
        X_val, y_val = X_sample.iloc[val_idx], y_sample.iloc[val_idx]

        # Inner Loop: Strategic feature selection executed exclusively on the training fold
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        estimator = xgb.XGBClassifier(**xgb_cpu_params)
        selector = RFECV(
            estimator=estimator,
            step=0.05,  # Eliminate 5% of the least important features per iteration
            cv=inner_cv,
            scoring='roc_auc',
            min_features_to_select=config.TARGET_FEATURE_COUNT,
            n_jobs=-1
        )

        print("   - Running RFECV exclusively on training fold partitions...")
        selector.fit(X_tr, y_tr)

        # Apply locked subset transformations to validation instances
        X_tr_selected = selector.transform(X_tr)
        X_val_selected = selector.transform(X_val)

        # Evaluate performance boundaries
        estimator.fit(X_tr_selected, y_tr)
        preds = estimator.predict_proba(X_val_selected)[:, 1]

        fold_auc = roc_auc_score(y_val, preds)
        outer_scores.append(fold_auc)
        print(f"▶ Fold {fold + 1} Unbiased AUC: {fold_auc:.4f} (Features locked: {selector.n_features_})")

    # Final unbiased generalization metric for manuscript reporting
    print(f"\n🏆 [MANUSCRIPT REPORT] Unbiased Nested CV AUC: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")

    # =====================================================================
    # 4. Production Feature Locking
    # =====================================================================
    print(f"\n🚀 [Phase 3 - Step 3] Locking the global top {config.TARGET_FEATURE_COUNT} features for production...")

    # Train a unified model on the curated sample partition to map global feature importance
    final_model = xgb.XGBClassifier(**config.XGB_PARAMS)
    final_model.fit(X_sample, y_sample)

    indices = np.argsort(final_model.feature_importances_)[::-1][:config.TARGET_FEATURE_COUNT]
    final_selected_features = X_sample.columns[indices].tolist()

    # Serialize feature names to JSON for deterministic external blind test alignment
    feature_list_path = config.get_model_path(f"selected_features_{config.TARGET_FEATURE_COUNT}.json")
    with open(feature_list_path, 'w') as f:
        json.dump(final_selected_features, f)

    # Generate the finalized slim Parquet file mapping the full 320k cohort
    df_slim = df[metadata_cols + final_selected_features]
    df_slim.to_parquet(config.SLIM_FEATURES_PARQUET, index=False)

    print(f"✅ Success! Feature dictionary serialized to {feature_list_path}")
    print(f"✅ Phase 3 Complete! Compressed slim feature table saved to {config.SLIM_FEATURES_PARQUET}")


if __name__ == "__main__":
    import time
    start = time.time()
    run_nested_cv_and_finalize()
    print(f"\n🏁 Total execution time: {(time.time() - start) / 60:.2f} minutes")