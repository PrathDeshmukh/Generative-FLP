import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import tight_layout
from sklearn.manifold import TSNE

from utils.utils import calcfps


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
    perplexity = 60

    if n_files != 0:
        fps_dict = {}
        for file in file_paths:
            fname = os.path.basename(file).split('.')[0]
            fps_list = calcfps(file)
            fps_dict[fname] = np.array(fps_list)
            print(f'Fingerprints for {fname} calculated!')

        fig, ax = plt.subplots(1,1, figsize=(12,12), tight_layout=True)
        palette = sns.color_palette("bright")
        label_colors = dict(zip(fps_dict.keys(), palette))

        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    init="pca",
                    random_state=42)

        for label, fps in fps_dict.items():
            embedding = tsne.fit_transform(fps)
            color = label_colors[label]
            sns.scatterplot(x=embedding[:,0], y=embedding[:,1], label=label, color=color, alpha=0.5)
            plt.legend()

        plt.title("t-SNE visualization of various datasets")
        plt.savefig(f'./Generative-FLP/data/tsne_p{perplexity}.png')
        print("Plot generated!")

    else:
        print("No files supplied!")