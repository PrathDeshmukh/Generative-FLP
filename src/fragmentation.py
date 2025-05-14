#Script for extracting the substituents from organocatalysts in oscar
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition
import pandas as pd
from rdkit.Chem import rdMolDescriptors as rdmd
import numpy as np

scaffold_csv = '/home/ppdeshmu/Generative-FLP/data/csv/scaffold_smiles_labelled.csv'
scaffold_df = pd.read_csv(scaffold_csv)

oscar_csv = '/home/ppdeshmu/Generative-FLP/data/csv/OSCAR_SMILES_all.csv'
oscar_df = pd.read_csv(oscar_csv)

scaf_mol_list = [Chem.MolFromSmiles(scaffold) for scaffold in scaffold_df['SMILES']]
osc_mol_list = [Chem.MolFromSmiles(oscar) for oscar in oscar_df['SMILES']]

groups_all = []
for scaf_mol in scaf_mol_list:
    rdgd = Chem.rdRGroupDecomposition.RGroupDecompositionParameters()
    rdgd.onlyMatchAtRGroups = True
    for osc_mol in osc_mol_list:
        try:
            groups, _ = rdRGroupDecomposition.RGroupDecompose([scaf_mol], [osc_mol], asSmiles=True, options=rdgd)
            if len(groups) != 0:
                del groups[0]['Core']
                groups_ = list(groups[0].values())
                groups_all.extend(groups_)
        except Exception:
            pass

#Remove duplicates by eliminating same SMILES strings
group_set = set(groups_all)

#Remove duplicates by comparing mol objects
subs_smiles = list(group_set)
scaf_og_csv = '/home/ppdeshmu/oscar_extraction/scaffold_smiles.csv'
scaf_og_df = pd.read_csv(scaf_og_csv)
scaf_og_mols = [Chem.MolFromSmiles(smi) for smi in scaf_og_df['SMILES']]

removed_smiles = []
kept_smiles = []

for smi in subs_smiles:
    subs_mol = Chem.MolFromSmiles(smi)
    matched = False
    for scaf_mol in scaf_og_mols:
        if subs_mol.HasSubstructMatch(scaf_mol):
            removed_smiles.append(smi)
            matched = True
            break
    if not matched:
        kept_smiles.append(smi)


#Removing outliers structures on the basis of molecular size by iqr method
nha_list = []
for smi in kept_smiles:
    mol = Chem.MolFromSmiles(smi)
    nha = rdmd.CalcNumHeavyAtoms(mol)
    nha_list.append(nha)

q1 = np.quantile(nha_list, 0.25)
q3 = np.quantile(nha_list, 0.75)
median = np.median(nha_list)

iqr = q3 - q1

upper_bound = q3 + 1.5*iqr
lower_bound = q1 - 1.5*iqr

cleaned_kept_smiles = []
for i, nha in enumerate(nha_list):
    if lower_bound < nha < upper_bound:
        cleaned_kept_smiles.append(kept_smiles[i])
    else:
        removed_smiles.append(kept_smiles[i])

print(len(cleaned_kept_smiles))

removed_df = pd.DataFrame({'SMILES':removed_smiles})
removed_df.to_csv('/home/ppdeshmu/Generative-FLP/data/csv/Removed_smiles.csv')

df = pd.DataFrame({'SMILES': cleaned_kept_smiles})
df.to_csv('/home/ppdeshmu/Generative-FLP/data/csv/Substituents_smiles.csv')