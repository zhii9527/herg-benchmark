import pandas as pd
import os
import sys
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D  # Import low-level drawing engine

# Ensure correct import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def draw_toxic_molecules():
    print("\n🚀 [Phase 6.3] Starting HIGH-RESOLUTION Pharmacophore Visualization...")

    if not os.path.exists(config.SLIM_FEATURES_PARQUET):
        print("❌ Error: Cannot find slim features table.")
        return

    df = pd.read_parquet(config.SLIM_FEATURES_PARQUET)

    # Filter 12 typical toxic molecules
    toxic_df = df[df['Label'] == 1].head(12)
    smiles_list = toxic_df['std_smiles'].tolist()

    mols = []
    highlight_atoms_list = []

    # Define key pharmacophore patterns driving hERG toxicity (e.g., basic nitrogen centers)
    toxic_smarts = Chem.MolFromSmarts("[N;!H0;v3,v4&+1]")

    print("   ▶ Processing molecules and identifying pharmacophores...")
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            matches = mol.GetSubstructMatches(toxic_smarts)
            hit_atoms = [atom for match in matches for atom in match]
            highlight_atoms_list.append(hit_atoms)

    if mols:
        pharma_dir = os.path.join(config.FIGURE_DIR, "pharmacophore_plots")
        os.makedirs(pharma_dir, exist_ok=True)

        # --- Core improvement: Set publication-level drawing parameters ---
        draw_options = Draw.MolDrawOptions()
        draw_options.bondLineWidth = 3                 # Thicken bonds (default is 2)
        draw_options.minFontSize = 14                  # Minimum atom font size
        draw_options.annotationFontScale = 1.0         # Annotation font scale
        draw_options.highlightBondWidthMultiplier = 4  # Make highlights more prominent

        # 1. Save as Ultra-HD PNG (suitable for presentations)
        print("   ▶ Saving Ultra-HD PNG...")
        img_png = Draw.MolsToGridImage(
            mols,
            molsPerRow=3,           # Reduce to 3 molecules per row for larger rendering
            subImgSize=(600, 600),  # Double the resolution
            highlightAtomLists=highlight_atoms_list,
            legends=[f"Toxic Molecule {i + 1}" for i in range(len(mols))],
            drawOptions=draw_options,
            returnPNG=False
        )
        img_png.save(os.path.join(pharma_dir, "Pharma_1_Toxic_Highlights_HD.png"))

        # 2. Save as SVG (Highly recommended for publications, infinitely scalable)
        print("   ▶ Saving Publication-quality SVG...")
        img_svg = Draw.MolsToGridImage(
            mols,
            molsPerRow=3,
            subImgSize=(600, 600),
            highlightAtomLists=highlight_atoms_list,
            legends=[f"Toxic Molecule {i + 1}" for i in range(len(mols))],
            drawOptions=draw_options,
            useSVG=True             # Enable vector image mode
        )
        with open(os.path.join(pharma_dir, "Pharma_1_Toxic_Highlights_Vector.svg"), "w") as f:
            f.write(img_svg)

        print(f"✅ Success! HD PNG and Vector SVG saved to: {pharma_dir}")
    else:
        print("⚠️ No suitable molecules found for drawing.")

if __name__ == "__main__":
    draw_toxic_molecules()