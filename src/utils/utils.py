#Script for generating dataset levels features, properties, fingerprints, etc.
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity, ExplicitBitVect


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

def calcIntDiv(fps_list: list) -> float:
    """
    Calculation of internal diversity of a dataset by Tanimoto distance metric
    Args:
        fps_list: Python list of fingerprints as bit vector objects

    Returns:
        intDiv: Internal diversity
    """
    nfps = len(fps_list)
    if fps_list and not isinstance(fps_list[0], ExplicitBitVect):
        raise TypeError("Excpected fingerprints as bit vector objects")

    dist_sum = 0
    count = 0

    for i in range(nfps):
        for j in range(i+1, nfps - 1):
            dist = TanimotoSimilarity(fps_list[i], fps_list[j], returnDistance=True)
            dist_sum += dist
            count += 1

    intDiv = dist_sum / count
    return intDiv