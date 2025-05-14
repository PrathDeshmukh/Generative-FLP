#Script for extracting the substituents from organocatalysts in oscar
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition
import pandas as pd

scaffold_csv = '/home/pratham/Desktop/data/scaffold_smiles.csv'
scaffold_df = pd.read_csv(scaffold_csv)

oscar_csv = '/home/pratham/Desktop/data/OSCAR_SMILES_all.csv'
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

removed_smiles = []
for smi in subs_smiles:
    subs_mol = Chem.MolFromSmiles(smi)
    for scaf_mol in scaf_mol_list:
        if subs_mol.HasSubstructMatch(scaf_mol):
            removed_smiles.append(smi)
            subs_smiles.remove(smi)
            break

removed_df = pd.DataFrame({'SMILES':removed_smiles})
removed_df.to_csv('/home/ppdeshmu/Generative-FLP/data/csv/Removed_smiles.csv')

df = pd.DataFrame({'SMILES': subs_smiles})
df.to_csv('/home/ppdeshmu/Generative-FLP/data/csv/Substituents_smiles.csv')