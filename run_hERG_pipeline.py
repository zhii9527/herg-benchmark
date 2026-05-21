import os
import time
import sys

# Ensure correct import path (pointing to src directory)
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import config

from src.pipelines import (
    step01_data_curation,
    step02_feature_pipeline,
    step03_feature_selection,
    step04_model_training,
    step05_model_evaluation
)

# --- NEW: Updated Imports for the Refactored Interpretation Module ---
from src.interpretation import (
    eda_report,
    model_benchmarking,
    plot_chemical_space,
    plot_shap_importance,
    pharmacophore_vis,
    generate_si_data
)

def main():
    print("===============================================================")
    print("🚀 hERG Toxicity Prediction Pipeline (Assay-Aware & Nested CV)")
    print("===============================================================")
    total_start_time = time.time()

    # ---------------------------------------------------------
    # Phase 1: Data Curation & Proxy Flagging
    # ---------------------------------------------------------
    print("\n>>> PHASE 1: Data Curation & Proxy Metrics")
    step01_data_curation.run_data_audit()

    # ---------------------------------------------------------
    # Phase 2: Feature Engineering (2048-bit Morgan + PhysChem)
    # ---------------------------------------------------------
    print("\n>>> PHASE 2: Feature Extraction")
    step02_feature_pipeline.extract_features()

    # ---------------------------------------------------------
    # Phase 3: Unbiased Feature Selection (Nested CV)
    # ---------------------------------------------------------
    print("\n>>> PHASE 3: Feature Selection & Collinearity Audit")
    step03_feature_selection.run_nested_cv_and_finalize()

    # ---------------------------------------------------------
    # Phase 4: Tiered Weighted Training
    # ---------------------------------------------------------
    print("\n>>> PHASE 4: Model Training (Scaffold Split & Weighted XGBoost)")
    step04_model_training.run_training()

    # ---------------------------------------------------------
    # Phase 5: External Validation & Statistical Testing
    # ---------------------------------------------------------
    print("\n>>> PHASE 5: Blind Test Validation & Bootstrap Statistics")
    step05_model_evaluation.run_external_validation()

    # ---------------------------------------------------------
    # Phase 6: Interpretability, Benchmarking & Visualization
    # ---------------------------------------------------------
    print("\n>>> PHASE 6: Generating Academic Figures & Benchmark Reports")
    # 1. Exploratory Data Analysis
    eda_report.run_herg_eda()
    # 2. Unified Benchmarking (ROC, PR, DeLong, Confusion Matrix)
    model_benchmarking.run_comprehensive_benchmark()
    # 3. Out-of-Distribution (OOD) Chemical Space t-SNE
    plot_chemical_space.generate_tsne_space_plot()
    # 4. Explainable AI (SHAP Summary)
    plot_shap_importance.generate_shap_summary_plot()
    # 5. High-Resolution Pharmacophore (Toxicophore) Highlighting
    pharmacophore_vis.draw_toxic_molecules()
    # 6. Supporting Information (5-Fold CV Robustness Engine)
    generate_si_data.run_si_generation()

    total_time = (time.time() - total_start_time) / 60
    print("\n===============================================================")
    print(f"🎉 Pipeline completed successfully in {total_time:.2f} minutes!")
    print(f"📁 Models saved to:  {config.MODEL_DIR}")
    print(f"📈 Figures saved to: {config.FIGURE_DIR}")
    print(f"📊 Reports saved to: {config.REPORT_DIR}") # Added report directory info
    print("===============================================================")


if __name__ == "__main__":
    main()