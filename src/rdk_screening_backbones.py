#Additional RDKit based screening of CSD extracted molecules
import os, glob
from rdkit import Chem
from rdkit.Chem import Descriptors
import pickle

pkl_files = glob.glob(os.path.join("./CSD_backbones", "*.pkl"))
total_smiles_count = 0
for pkl_f in pkl_files:
    with open(pkl_f, 'rb') as f:
        file_dict = pickle.load(f)
        f.close()

    smiles = file_dict['SMILES']

    filtered_smiles = []
    reject_count = 0
    for smi in smiles:
        try:
            #Check for unphysical bonding and radical electrons in the molecule
            if Descriptors.NumRadicalElectrons(Chem.MolFromSmiles(smi)) > 0:
                reject_count += 1
                print(f"Radicals detected in SMILES: {smi}")
            else:
                filtered_smiles.append(smi)

        except Exception as e:
            reject_count += 1

    total_smiles_count += len(filtered_smiles)

    print(f'Accepted SMILES: {len(filtered_smiles)}')
    print(f'Rejected SMILES: {reject_count}')

    with open(pkl_f, 'wb') as f:
      pickle.dump({'SMILES': filtered_smiles}, f)
      f.close()

print(f"Total SMILES after screening: {total_smiles_count}")