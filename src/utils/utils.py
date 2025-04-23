#Script for generating dataset levels features, properties, fingerprints, etc.
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles


def calcfps(dataset: str,
            radius: int=2,
            fps_size: int=2048,
            bitVectObj: bool=False) -> list:
    """
    Calculates the fingerprints of a given dataset of SMILES
    Args:
         dataset: CSV file containing a column titled "SMILES"
         radius: fingerprint radius
         fps_size: vector length of fingerprint
         bitVectObj: Whether to return the fingerprint as a bit vector object or a list
    Returns:
        fps_list: List of fingerprints as bit vector
    """
    df = pd.read_csv(dataset)
    smiles = df['SMILES']
    mols = []
    for smi in smiles:
        try:
            mol = MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
        except Exception as e:
            print(e)

    fpsgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fps_size)
    fps_list = []
    for mol in mols:
        try:
            fps = fpsgen.GetFingerprint(mol)
            if bitVectObj is False:   #Converts the fingerprint vector to Python list
                fps = fps.ToList()
            fps_list.append(fps)

        except Exception as e:
            print(e)

    print(f"Total SMILES in the dataset: {len(smiles)}")
    print(f"Total fingerprints calculated: {len(fps_list)}")

    return fps_list