#Gather SMILES of FLPs from derived and screened from CSD
import pickle
import glob
import os
from rdkit import Chem
import pandas as pd

bkb_path = './CSD_backbones'
pkl_files = glob.glob(os.path.join(bkb_path, '*.pkl'))

all_smiles = []
flp = []
for p_file in pkl_files:
    with open(p_file, 'rb') as f:
        p_dict = pickle.load(f)
        f.close()

    smiles = p_dict['SMILES']
    all_smiles.extend(smiles)
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        B_count = 0
        N_count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'B':
                B_count += 1
            if atom.GetSymbol() == 'N':
                N_count += 1
        if B_count == 1 and N_count <= 3:
            flp.append(s)

print(f'Total number of SMILES: {len(all_smiles)}')
print(f'Screened SMILES: {len(set(flp))}')

df = pd.DataFrame({'SMILES': list(set(flp))})
df.to_csv('./one_B_three_N.csv')
