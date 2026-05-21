import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit import RDLogger
from tqdm import tqdm

# Ensure global configuration is imported (adjust path if the directory hierarchy changes)
import config

# Suppress RDKit warning messages to minimize console noise
RDLogger.DisableLog('rdApp.*')


def check_pains(smi, catalog):
    """Checks for Pan-Assay Interference Compounds (PAINS) structural alerts."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol and catalog.HasMatch(mol):
            return 'low_confidence'
        return 'high_confidence'
    except:
        return 'low_confidence'


def generate_inchikey(smi):
    """Generates a unique InChIKey hash identifier for a given molecule."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToInchiKey(mol)
        return None
    except:
        return None


def run_data_audit():
    """
    Pipeline Phase 1: Execute Data Quality Audit and Confidence Annotation
    """
    print(f"\n🚀 [Phase 1] Starting Data Curation & Proxy Flagging...")

    # 1. Load raw dataset
    if not os.path.exists(config.RAW_CSV_INPUT):
        print(f"❌ Error: Cannot find raw data {config.RAW_CSV_INPUT}")
        return

    df = pd.read_csv(config.RAW_CSV_INPUT)
    print(f"▶ Initial records loaded: {len(df)}")

    # If curated data already exists, skip to save time (breakpoint resumption support)
    if os.path.exists(config.CLEANED_RECORDS_CSV):
        print(f"✨ [Skip] Cleaned records already exist at {config.CLEANED_RECORDS_CSV}.")
        return

    # 2. Generate InChIKeys (The foundation for cross-database alignment)
    print("▶ Generating InChIKeys for structural cross-referencing...")
    tqdm.pandas(desc="InChIKey Generation")
    df['inchikey'] = df['std_smiles'].progress_apply(generate_inchikey)

    # 3. Structural Alert Scanning (PAINS Filtering)
    print("▶ Flagging structural alerts (PAINS)...")
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    tqdm.pandas(desc="PAINS Scanning")
    df['pains_flag'] = df['std_smiles'].progress_apply(lambda x: check_pains(x, catalog))
    print(f"▶ Detected {sum(df['pains_flag'] == 'low_confidence')} molecules with PAINS alerts.")

    # 4. Multi-Source Consensus Audit (Cross-Database Label Matching)
    if os.path.exists(config.EXTERNAL_CONSENSUS_DATA):
        print(f"▶ Loading consensus reference: {config.EXTERNAL_CONSENSUS_DATA}")
        df_ext = pd.read_csv(config.EXTERNAL_CONSENSUS_DATA)

        # Assume molecules with IC50 < 10000 nM are toxic (1), otherwise safe (0) in the reference set
        df_ext['Label_ext'] = (df_ext['IC50_nM'] < 10000).astype(int)

        # Merge datasets using left join on InChIKey
        merged = df.merge(df_ext[['inchikey', 'Label_ext']], on='inchikey', how='left')

        df['consensus_flag'] = 'high_confidence'

        # Conflict detection: Low confidence if external labels contradict the primary training labels
        mask_conflict = merged['Label_ext'].notna() & (merged['Label'] != merged['Label_ext'])
        df.loc[mask_conflict, 'consensus_flag'] = 'low_confidence'

        print(
            f"▶ Consensus audit complete. Low confidence (Conflicts) flagged: {sum(df['consensus_flag'] == 'low_confidence')}")
    else:
        df['consensus_flag'] = 'high_confidence'
        print("⚠️ Warning: External consensus data not found. Skipping cross-reference.")

    # 5. Aggregate Confidence Metrics (Strict Single-Vote Disqualification Strategy)
    df['final_conf_flag'] = 'high_confidence'
    df.loc[(df['pains_flag'] == 'low_confidence') | (
                df['consensus_flag'] == 'low_confidence'), 'final_conf_flag'] = 'low_confidence'

    # 6. Serialize and Export Final Output
    df.to_csv(config.CLEANED_RECORDS_CSV, index=False)
    print(f"✅ Phase 1 Complete! Cleaned records saved to: {config.CLEANED_RECORDS_CSV}")
    print(
        f"📊 Final summary: {sum(df['final_conf_flag'] == 'high_confidence')} High vs {sum(df['final_conf_flag'] == 'low_confidence')} Low confidence.")


if __name__ == "__main__":
    # Allow standalone execution for pipeline validation
    run_data_audit()