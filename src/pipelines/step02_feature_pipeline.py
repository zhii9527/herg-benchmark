import pandas as pd
import numpy as np
import os
import time
import gc
import sys

# Ensure parent directory is accessible for global configuration imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
from tqdm import tqdm
from rdkit.Chem import MACCSkeys

# Suppress RDKit warning messages to minimize console noise
RDLogger.DisableLog('rdApp.*')


def extract_features():
    """Pipeline Phase 2: 2D Morgan(R2, 2048) -> MACCS -> PhysChem -> Fusion"""
    print(f"\n🚀 [Phase 2] Starting Feature Extraction Pipeline...")

    final_output = config.FULL_FEATURES_PARQUET
    raw_input = config.CLEANED_RECORDS_CSV

    if not os.path.exists(raw_input):
        print(f"❌ Error: Cannot find curated data at {raw_input}. Please run 01_data_curation.py first.")
        return

    # --- 🚀 Intelligent Skip Logic (Prevents redundant calculation during full pipeline execution) ---
    if os.path.exists(final_output):
        print(f"✨ [Skip] Master feature matrix already exists at {final_output}. Delete the file manually to re-extract.")
        return

    print(f"▶ Loading curated dataset: {len(pd.read_csv(raw_input))} records found.")
    df_base = pd.read_csv(raw_input)
    smiles_list = df_base['std_smiles'].tolist()

    # --- Phase 1: Iterative Descriptor Calculation ---
    # 💡 Core Configuration: Radius=2, fpSize=2048
    radius = 2
    fp_size = 2048
    print(f"▶ Initializing Morgan Fingerprint Generator (Radius={radius}, Size={fp_size})...")
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    desc_names = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    morgans, maccs_list, physchems = [], [], []
    for smi in tqdm(smiles_list, desc="Core Feature Calculation"):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # 1. Morgan Fingerprints (2048 dimensions)
            morgans.append(mfpgen.GetFingerprintAsNumPy(mol).astype(np.int8))
            # 2. MACCS Keys (166 structural keys)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            maccs_list.append([int(maccs_fp[i]) for i in range(1, 167)])
            # 3. Physicochemical Descriptors
            physchems.append(list(calc.CalcDescriptors(mol)))
        else:
            morgans.append(np.zeros(fp_size, dtype=np.int8))
            maccs_list.append([0] * 166)
            physchems.append([np.nan] * len(desc_names))

    # --- Phase 2: Memory-Optimized Conversion ---
    print("\n▶ Executing memory-optimized DataFrame conversion (Ensure adequate physical RAM)...")
    X_2d = pd.DataFrame(np.array(morgans), columns=[f"Morgan_{i}" for i in range(fp_size)])
    del morgans

    X_maccs = pd.DataFrame(np.array(maccs_list, dtype=np.int8), columns=[f"MACCS_{i}" for i in range(1, 167)])
    del maccs_list

    X_phys = pd.DataFrame(physchems, columns=desc_names).astype(np.float32)
    X_phys.fillna(X_phys.median(), inplace=True)
    del physchems

    gc.collect()  # Explicit garbage collection to free runtime memory allocation

    # --- Phase 3: Feature Fusion and Serialization ---
    print("▶ Executing feature matrix fusion...")
    df_final = pd.concat([df_base.reset_index(drop=True), X_2d, X_maccs, X_phys], axis=1)

    # Export to compressed Parquet format
    df_final.to_parquet(final_output, index=False, engine='pyarrow')
    print(f"\n✅ Phase 2 Complete! Feature engineering finalized. Matrix shape: {df_final.shape}")
    print(f"📁 Master feature table saved to: {final_output}")


if __name__ == "__main__":
    start = time.time()
    extract_features()
    print(f"🏁 Total execution time: {(time.time() - start) / 60:.2f} minutes")