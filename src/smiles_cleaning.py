import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker


def parse_cmd():
    parser = argparse.ArgumentParser(description="Cleaning of assembled SMILES and selecting the most diverse SMILES for dataset generation")

    parser.add_argument(
        "--smiles_path",
        type=str,
        default="/home/ppdeshmu/Generative-FLP/data/assembled-smiles-1000.csv"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/ppdeshmu/Generative-FLP/data"
    )

    return parser.parse_args()


cmd_args = parse_cmd()
smiles_path = cmd_args.smiles_path
save_path = cmd_args.save_path

smiles_df = pd.read_csv(smiles_path)

#canonicalize SMILES
successful_canons = 0
unsuccessful_canons = 0
canonicalized_smiles = []

for smi in smiles_df['SMILES'].to_list():
    try:
        canon_smi = Chem.CanonSmiles(smi)
        successful_canons += 1
        canonicalized_smiles.append(canon_smi)
    except Exception as e:
        print(f"Canonicalization failed for SMILES:{smi}")
        unsuccessful_canons += 1

print("Canonicalization complete!")
print(f"Successful canonicalizations:{successful_canons}\n")

#Checking for duplicates by converting list to set and back
canonicalized_smiles = list(set(canonicalized_smiles))

# Dissimilarity check and pick using fingerprints as representation and Tanimoto as similarity metric
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in canonicalized_smiles]
nfps = len(fps)

picker = MaxMinPicker()
pickIndices = list(picker.LazyBitVectorPick(fps, nfps, 500, seed=42))

picks = [canonicalized_smiles[pi] for pi in pickIndices]

print("Diversity based subset created!")

flp_smiles = pd.DataFrame(picks, columns=['SMILES'])
flp_smiles.to_csv(f'{save_path}FLP_smiles.csv')

print(f"Subset saved at location {save_path}")