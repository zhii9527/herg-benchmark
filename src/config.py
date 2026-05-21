import os

# =================================================================
# 1. Base Path Configuration
# =================================================================
BASE_DIR = r"E:\6890ML\ML\hERG_Toxicity_Project"

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

RESULT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
REPORT_DIR = os.path.join(RESULT_DIR, "reports")
MODEL_DIR = os.path.join(RESULT_DIR, "models")

# Ensure all required directories are created initialization-phase
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULT_DIR, FIGURE_DIR, REPORT_DIR, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

# =================================================================
# 2. Specific File Paths
# =================================================================
RAW_CSV_INPUT = os.path.join(RAW_DATA_DIR, "hERG_original_320k.csv")
FULL_FEATURES_PARQUET = os.path.join(PROCESSED_DATA_DIR, "hERG_Full_Features.parquet")
SLIM_FEATURES_PARQUET = os.path.join(PROCESSED_DATA_DIR, "hERG_Slim_250_Features.parquet")

# Curated records annotated with data proxy flags and confidence scores
CLEANED_RECORDS_CSV = os.path.join(PROCESSED_DATA_DIR, "hERG_Cleaned_with_Proxy.csv")
# External reference data used for multi-source consensus filtering
EXTERNAL_CONSENSUS_DATA = os.path.join(RAW_DATA_DIR, "external_chembl_pubchem_consensus.csv")

# =================================================================
# 3. Model Hyperparameters (Optimized XGBoost)
# =================================================================
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'max_depth': 3,               # Keep depth shallow to limit complexity
    'learning_rate': 0.05,        # Step size shrinkage
    'n_estimators': 1000,         # Total number of boosting rounds
    'subsample': 0.5,             # Row subsample ratio of training instances
    'colsample_bytree': 0.3,      # Column subsample ratio when constructing each tree
    'reg_alpha': 1.0,             # L1 regularization term on weights
    'reg_lambda': 50.0,           # Strong L2 regularization (critical for overfit prevention)
    'min_child_weight': 10,
    'gamma': 0.5,                 # Minimum loss reduction required to make a split
    'n_jobs': -1,
    'random_state': 42,
    'seed': 42,
    'device': 'cuda',
    'tree_method': 'hist'
}

# =================================================================
# 4. Task Logic Thresholds and SHAP Configurations
# =================================================================
CORR_THRESHOLD = 0.9
TARGET_FEATURE_COUNT = 250
AD_PERCENTILE = 95
SHAP_INTERACTION_PAIRS = ("fr_piperdine", "MolLogP")
SHAP_CONTRADICTION_FEATURE = "Morgan_1791"

# =================================================================
# 5. Data Auditing and Sample Weighting (New: Professor's Suggestions)
# =================================================================
BASE_WEIGHTS = {'ChEMBL': 1.0, 'PubChem': 0.8, 'hERGCentral': 0.5}
CONFIDENCE_WEIGHTS = {'high_confidence': 1.0, 'low_confidence': 0.3}
IC50_LIMITS = (1.0, 100000.0)

# =================================================================
# 6. Auxiliary Utility Functions
# =================================================================
def get_report_path(filename):
    return os.path.join(REPORT_DIR, filename)

def get_figure_path(filename):
    return os.path.join(FIGURE_DIR, filename)

def get_model_path(filename):
    return os.path.join(MODEL_DIR, filename)

if __name__ == "__main__":
    print(f"✅ Config loaded. Project Root: {BASE_DIR}")
    print(f"📊 Processed Data Directory: {PROCESSED_DATA_DIR}")
    print(f"🔍 Audit Data Path: {CLEANED_RECORDS_CSV}")