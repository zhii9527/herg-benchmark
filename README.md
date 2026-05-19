# hERG Toxicity Prediction & Chemical Space Analysis

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green.svg)

## Overview
This repository hosts an advanced machine learning pipeline for predicting **hERG (human Ether-à-go-go-Related Gene) channel inhibition**, a critical anti-target in drug discovery. The project focuses on rigorous out-of-distribution (OOD) validation, high-quality pharmacophore visualization, and statistical benchmarking against standard and state-of-the-art models.

## Key Features
* **Assay-Weighted Tree Ensembles:** Implements confidence-weighted XGBoost algorithms for imbalanced/noisy bioactivity data.
* **Rigorous Validation:** Evaluates performance on a temporally and structurally distinct external validation set (ChEMBL blind test).
* **Statistical Significance:** Integrates true DeLong ROC tests to validate model superiority objectively.
* **Explainable AI (XAI):** Features comprehensive SHAP (SHapley Additive exPlanations) analysis and high-resolution RDKit pharmacophore highlighting for toxicophore identification.
* **Chemical Space Projection:** Employs advanced t-SNE + KDE visualization to map the distribution shifts between training and OOD datasets.

## Repository Structure
\`\`\`text
├── pipelines/               # Core data processing and training pipelines
├── interpretation/          # Downstream evaluation and plotting scripts
│   ├── eda_report.py
│   ├── model_benchmarking.py
│   ├── plot_chemical_space.py
│   ├── plot_shap_importance.py
│   └── pharmacophore_vis.py
├── requirements.txt         # Environment dependencies
└── README.md
\`\`\`

## Installation

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/your-username/herg-toxicity-predictor.git
   cd herg-toxicity-predictor
   \`\`\`

2. Create a virtual environment and install dependencies:
   \`\`\`bash
   conda create -n herg_env python=3.9
   conda activate herg_env
   pip install -r requirements.txt
   \`\`\`

## Usage
All downstream evaluation scripts are located in the \`interpretation\` directory. Once you have executed the data extraction and training pipelines (generating the necessary \`.parquet\` and \`.joblib\` files), you can run the benchmarking tools:

\`\`\`bash
# Run the comprehensive benchmark (ROC/PR curves, DeLong test, Confusion Matrices)
python interpretation/model_benchmarking.py

# Generate Chemical Space t-SNE projections
python interpretation/plot_chemical_space.py

# Visualize high-resolution toxicophores
python interpretation/pharmacophore_vis.py
\`\`\`

## License
This project is licensed under the MIT License - see the LICENSE file for details.
