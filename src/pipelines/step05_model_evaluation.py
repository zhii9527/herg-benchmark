import pandas as pd
import numpy as np
import os
import sys
import joblib
import json
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, \
    average_precision_score, matthews_corrcoef
from sklearn.utils import resample
from rdkit import RDLogger

# Ensure parent directory is accessible for global configuration imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Suppress RDKit warning messages to minimize console noise
RDLogger.DisableLog('rdApp.*')


def extract_features_for_new_data(smiles_list, selected_features):
    """Extracts model-aligned feature subsets for new chemical entities to match the training feature space."""
    print(f"   ▶ Extracting features for {len(smiles_list)} future molecules...")

    # 1. Initialize generators (Must strictly match training parameters: Radius=2, fpSize=2048)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    desc_names = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    all_feat_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            all_feat_list.append({f: 0 for f in selected_features})
            continue

        # Compute 2D structural fingerprints and physicochemical descriptors
        morgan_fp = mfpgen.GetFingerprintAsNumPy(mol)
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        phys_vals = list(calc.CalcDescriptors(mol))

        feat_dict = {}
        for i, val in enumerate(morgan_fp): feat_dict[f"Morgan_{i}"] = val
        for i in range(1, 167): feat_dict[f"MACCS_{i}"] = int(maccs_fp[i])
        for i, name in enumerate(desc_names): feat_dict[name] = phys_vals[i]

        all_feat_list.append({f: feat_dict.get(f, 0) for f in selected_features})

    X = pd.DataFrame(all_feat_list).fillna(0)
    return X


def compute_bootstrap_intervals(y_true, y_prob, threshold=0.2, n_iterations=1000):
    """Computes 95% confidence intervals (CIs) for evaluation metrics via bootstrap resampling."""
    metrics = {'ROC-AUC': [], 'PR-AUC': [], 'F1 (@0.2)': [], 'MCC (@0.2)': []}
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    for i in range(n_iterations):
        indices = resample(np.arange(len(y_true)), random_state=i)
        if len(np.unique(y_true[indices])) < 2:
            continue

        metrics['ROC-AUC'].append(roc_auc_score(y_true[indices], y_prob[indices]))
        metrics['PR-AUC'].append(average_precision_score(y_true[indices], y_prob[indices]))
        metrics['F1 (@0.2)'].append(f1_score(y_true[indices], y_pred[indices]))
        metrics['MCC (@0.2)'].append(matthews_corrcoef(y_true[indices], y_pred[indices]))

    results = {}
    for m in metrics:
        lower = np.percentile(metrics[m], 2.5)
        upper = np.percentile(metrics[m], 97.5)
        mean = np.mean(metrics[m])
        results[m] = f"{mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})"
    return results


def run_external_validation():
    """Pipeline Phase 5: External Validation and Uncertainty Quantification via Bootstrapping."""
    print(f"\n🚀 [Phase 5] Starting External Validation & Statistical Testing...")

    model_path = config.get_model_path("hERG_weighted_xgb_model.joblib")
    feature_path = config.get_model_path(f"selected_features_{config.TARGET_FEATURE_COUNT}.json")

    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        print(f"❌ Error: Model artifact or feature list configuration not found. Please execute training and selection pipelines first.")
        return

    model = joblib.load(model_path)
    with open(feature_path, 'r') as f:
        selected_features = json.load(f)

    # Isolated temporal blind validation set (e.g., ChEMBL 2022-2024)
    future_csv = os.path.join(config.RAW_DATA_DIR, "ChEMBL36_Future_Validation.csv")
    if not os.path.exists(future_csv):
        print(f"❌ Error: External blind validation dataset not found at {future_csv}.")
        return

    df_future = pd.read_csv(future_csv)
    X_future = extract_features_for_new_data(df_future['canonical_smiles'], selected_features)
    y_true = df_future['Label']

    # Evaluate validation metrics at the optimized empirical threshold of 0.20
    threshold = 0.20
    y_prob = model.predict_proba(X_future)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc_val = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("\n" + "🌍" * 20)
    print("  FUTURE VALIDATION REPORT (ChEMBL 2022-2024)")
    print("🌍" * 20)
    print(f"Total Samples:     {len(df_future)}")
    print(f"ROC-AUC:           {auc_val:.4f}")
    print(f"F1-Score:          {f1:.4f} (at empirical threshold {threshold})")
    print(f"Recall:            {rec:.4f} (Sensitivity)")
    print(f"Precision:         {prec:.4f} (PPV)")
    print(f"Confusion Matrix:  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("=" * 40)

    print("\n   ⚖️ Running 1000x Bootstrap Resampling for 95% Confidence Intervals...")
    intervals = compute_bootstrap_intervals(y_true, y_prob, threshold=threshold)
    for k, v in intervals.items():
        print(f"   - {k}: {v}")
    print(f"\n✅ Phase 5 Complete! You can report these CI metrics in your manuscript.")


if __name__ == "__main__":
    import time

    start = time.time()
    run_external_validation()
    print(f"\n🏁 Total execution time: {(time.time() - start) / 60:.2f} minutes")