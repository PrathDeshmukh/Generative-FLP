import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rdkit.Chem
import seaborn as sns
from sklearn.manifold import TSNE


from rdkit.Chem.rdmolfiles import MolFromSmiles
from rdkit.Chem import rdFingerprintGenerator

def moltoFPS(mol: rdkit.Chem.Mol):
    try:
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps1 = fpgen.GetFingerprint(mol)
        return list(fps1)

    except:
        print('Exception!')

def smilestoFPS(smiles_csv) -> list:
    df = pd.read_csv(smiles_csv).head(200)
    smiles = df['SMILES']
    fps2 = []
    for smi in smiles:
        try:
            mol = MolFromSmiles(smi)
            fp = moltoFPS(mol)
            if fp is not None:
                fps2.append(fp)
        except:
            print('Exception')
    return fps2

def inputtoFPS(input_path: str, fps_dict: dict):
    name = os.path.basename(input_path).split('.')[0]
    print(name)
    fps_ = np.array(smilestoFPS(input_path))
    fps_dict[name] = fps_
    return fps_dict

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
            print(f'{file} done')

        tsne = TSNE(n_components=2, perplexity=30, init="pca")
        for label, fps in fps_dict.items():
            embedding = tsne.fit_transform(fps)
            sns.scatterplot(x=embedding[:,0], y=embedding[:,1], label=label)
            plt.legend()

        plt.savefig('./test_tsne.png')
        print("Plot generated!")

    else:
        print("No files supplied!")