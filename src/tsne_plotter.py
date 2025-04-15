import argparse
import os, glob
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rdkit.Chem
import seaborn as sns
from sklearn.manifold import TSNE

import ase
from ase.io import read

from rdkit.Chem.rdmolfiles import MolFromXYZBlock, MolFromSmiles
from rdkit.Chem import rdFingerprintGenerator


def atoms_to_xyz_str(atoms: ase.Atoms):
  f = StringIO()
  atoms.write(f, format="xyz")
  return f.getvalue()

def moltoFPS(mol: rdkit.Chem.Mol):
    try:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        fps = fpgen.GetFingerprint(mol)
        return fps
    except:
        print('Exception!')

def xyztoFPS(xyz):
    atoms = read(xyz)
    mol = MolFromXYZBlock(atoms_to_xyz_str(atoms))
    fps = moltoFPS(mol)
    return fps

def smilestoFPS(smiles_csv) -> list:
    df = pd.read_csv(smiles_csv)
    smiles = df['SMILES']
    fps = [moltoFPS(MolFromSmiles(smi)) for smi in smiles]
    return fps

def inputtoFPS(input_path: str, fps_dict: dict):
    fps = []
    if os.path.isdir(input_path):
        name = os.path.basename(input_path)
        xyz = glob.glob(os.path.join(input_path, '*.xyz'))
        for f in xyz:
            fp = xyztoFPS(f)
            fps.append(fp)
        fps_dict[name] = np.array(fps)
        return fps_dict

    elif os.path.isfile(input_path):
        name = os.path.basename(input_path)[0]
        fps.append(smilestoFPS(input_path))
        fps_dict[name] = np.array(fps)
        return fps_dict

    else:
        print(f"Input {input_path} is neither file nor file path")

if __name__ == '__main__':
    def parse_cmd():
        parser = argparse.ArgumentParser(description="Generate t-SNE plots of various databases")

        parser.add_argument(
            "--file_paths",
            nargs='*'
        )

        return parser.parse_args()

    cmd_args = parse_cmd()
    file_paths = cmd_args.file_paths

    if len(file_paths)!=0:
        fps_dict = {}
        for file in file_paths:
            fps_dict = inputtoFPS(file, fps_dict)

        tsne = TSNE(n_components=2, perplexity=30, init="pca")
        for label, fps in fps_dict.items():
            embedding = tsne.fit_transform(fps)
            sns.scatterplot(x=embedding[:,0], y=embedding[:,1], label=label)

        plt.legend()
        plt.savefig('./test_t-sne.png')
        print("Plot generated")

    else:
        print("No files supplied!")