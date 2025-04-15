import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from statistics import mean

path = '/home/ppdeshmu/Generative-FLP/data/FLP_smiles.csv'
smiles_csv = pd.read_csv(path)
smiles = smiles_csv['SMILES']

mols = [Chem.MolFromSmiles(smi) for smi in smiles.values]

#Molecular sizes plot
mol_sizes = [mol.GetNumAtoms() for mol in mols]

#Plot mol_sizes histogram
ax = sns.histplot(mol_sizes)
ax.set_title('Distribution of molecule sizes')
ax.set(xlabel='Number of atoms', ylabel='Frequency')
plt.savefig('/home/ppdeshmu/Generative-FLP/data/Molecular_size_dist.png')
plt.close()

#Diversity estimation
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
fps = [fpgen.GetFingerprint(mol) for mol in mols]

similarity = []
for n in range(len(fps)):
    s1 = BulkTanimotoSimilarity(fps[n], fps[n+1:])
    s2 = BulkTanimotoSimilarity(fps[n], fps[:n])
    s = s1 + s2
    mean_s = mean(s)
    similarity.append(mean_s)
print(len(similarity))
ax = sns.histplot(similarity)
ax.set_title('Distribution of similarity')
ax.set(xlabel='Tanimoto similarity', ylabel='Frequency')

plt.savefig('/home/ppdeshmu/Generative-FLP/data/diversity_dis.png')