from traceback import print_tb

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from utils import calcfps
import pandas as pd

path = "./assembled_FLP.csv"
df = pd.read_csv(path)
smiles = df['SMILES']

fps = calcfps(dataset=path)

kmeans = KMeans(n_clusters=50, random_state=42).fit(fps)

cluster_ctrs = kmeans.cluster_centers_

closest_idx, _ = pairwise_distances_argmin_min(cluster_ctrs, fps)
print(f"Number of cluster centers {len(closest_idx)}")

cluster_ctr_smiles = {'SMILES':[]}

for idx in closest_idx:
    cluster_ctr_smiles['SMILES'].append(smiles[idx])

cluster_smiles_df = pd.DataFrame(cluster_ctr_smiles)
cluster_smiles_df.to_csv('./cluster_center_smiles.csv')