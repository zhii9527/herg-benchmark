import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import config
from rdkit.Chem import Descriptors, rdFingerprintGenerator, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors

def generate_scaffold(smiles):
    """Generates the Bemis-Murcko scaffold for a given SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.GetScaffoldForMol(mol) if mol else ""
    except:
        return ""


def scaffold_split(df, train_frac=0.8, random_state=42):
    """Performs Bemis-Murcko scaffold-based splitting to prevent structural data leakage."""
    print("🧬 [Split] Generating Murcko Scaffolds for data partitioning...")
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
    """Applies a tiered sample weighting logic optimized for multi-source TDC data data integration."""
    print("⚖️ [Weighting] Applying TDC-specific layered weights...")

    def get_base_weight(source_str):
        source_str = str(source_str).upper()
        if 'CHEMBL' in source_str: return 1.0
        if 'KARIM' in source_str: return 0.8
        if 'CENTRAL' in source_str: return 0.5
        if 'TDC' in source_str: return 0.6
        return 0.5

    s_weights = df['Source'].apply(get_base_weight)
    c_weights = df['final_conf_flag'].map(config.CONFIDENCE_WEIGHTS).fillna(0.3)
    final_weights = s_weights * c_weights

    print(
        f"▶ Weight Summary: Max={final_weights.max():.2f}, Min={final_weights.min():.2f}, Mean={final_weights.mean():.3f}")
    return final_weights


def extract_features_for_new_data(smiles_list, selected_features):
    """Extracts model-aligned 250-dimensional features for external inference (Radius=2, Size=2048)."""
    print(f"🧬 Extracting features for {len(smiles_list)} future molecules...")

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    desc_names = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    all_feat_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            all_feat_list.append({f: 0 for f in selected_features})
            continue

        morgan_fp = mfpgen.GetFingerprintAsNumPy(mol)
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        phys_vals = list(calc.CalcDescriptors(mol))

        feat_dict = {}
        for i, val in enumerate(morgan_fp): feat_dict[f"Morgan_{i}"] = val
        for i in range(1, 167): feat_dict[f"MACCS_{i}"] = int(maccs_fp[i])
        for i, name in enumerate(desc_names): feat_dict[name] = phys_vals[i]

        all_feat_list.append({f: feat_dict.get(f, 0) for f in selected_features})

    return pd.DataFrame(all_feat_list).fillna(0)