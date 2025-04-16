import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rdkit.Chem
import seaborn as sns
from matplotlib.pyplot import tight_layout
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
    df = pd.read_csv(smiles_csv)
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

    n_files = len(file_paths)

    if n_files != 0:
        fps_dict = {}
        for file in file_paths:
            fps_dict = inputtoFPS(file, fps_dict)
            print(f'{file} done')

        fig, ax = plt.subplots(1,1, figsize=(12,12), tight_layout=True)
        palette = sns.color_palette("bright")
        label_colors = dict(zip(fps_dict.keys(), palette))

        tsne = TSNE(n_components=2,
                    perplexity=60,
                    init="pca",
                    random_state=42)

        for label, fps in fps_dict.items():
            embedding = tsne.fit_transform(fps)
            color = label_colors[label]
            sns.scatterplot(x=embedding[:,0], y=embedding[:,1], label=label, color=color, alpha=0.5)
            plt.legend()

        plt.title("t-SNE visualization of various datasets")
        plt.savefig('./test-tsne.png')
        print("Plot generated!")

    else:
        print("No files supplied!")