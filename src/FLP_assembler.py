import random

from rdkit import Chem
from rdkit.Chem import rdchem
from functools import reduce
from collections import defaultdict
import pandas as pd

#Get index of atoms that are to be connected
def get_connectivity_atom_idx(smiles):
    atom_indices = []

    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            neighbors = atom.GetNeighbors()
            atom_indices.extend([n.GetIdx() for n in neighbors])
    return atom_indices

def set_connectivity_tags(smiles):

    mol = Chem.MolFromSmiles(smiles)
    idx = get_connectivity_atom_idx(smiles)

    for atom in mol.GetAtoms():
        if atom.GetIdx() in idx:
            if atom.GetSymbol() in ['B']:
                atom.SetProp('ConnectType', 'LA_Backbone')
            elif atom.GetSymbol() in ['N']:
                atom.SetProp('ConnectType', 'LB_Backbone')
            else:
                atom.SetProp('ConnectType', 'Substituent')

        else:
            atom.SetProp('ConnectType', 'noConnect')
    return mol

#Remove dummy atoms '*' which appear after bond breaking during FLP fragmentation
def remove_dummy_atoms(mol):
    dummy_idx = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']

    rwmol = rdchem.RWMol(mol)
    for d_idx in reversed(dummy_idx):
        rwmol.RemoveAtom(d_idx)

    return rwmol.GetMol()

#Get the indexes of atoms that are to be connected in the combined mol object
def get_tagged_idx(mol):
    idx_dict = defaultdict(list)

    for atom in mol.GetAtoms():
        idx_dict[atom.GetProp('ConnectType')].append(atom.GetIdx())

    return idx_dict

def assembler(backbone, substituents_dict):

    clean_bb = remove_dummy_atoms(set_connectivity_tags(backbone))

    #Substitute LA first
    clean_subs_LA = [remove_dummy_atoms(set_connectivity_tags(substituent)) for substituent in substituents_dict['LA']]

    combo = reduce(Chem.CombineMols, [clean_bb, *clean_subs_LA])
    editable_raw_mol = Chem.EditableMol(combo)

    c_mol = editable_raw_mol.GetMol()

    tags = get_tagged_idx(c_mol)

    la_bb = tags['LA_Backbone']
    sub_tags = tags.get('Substituent', [])

    for sub in sub_tags:
        editable_raw_mol.AddBond(la_bb[0], sub, order=Chem.rdchem.BondType.SINGLE)

    editable_mol = editable_raw_mol.GetMol()

    #Clear property tags to prevent bonding the same atoms again
    for atom in editable_mol.GetAtoms():
        if atom.GetProp('ConnectType') in ['LA_Backbone', 'Substituent']:
            atom.SetProp('ConnectType', 'noConnect')


    #Substitute the LB second
    clean_subs_LB = [remove_dummy_atoms(set_connectivity_tags(substituent)) for substituent in substituents_dict['LB']]
    combo_2 = reduce(Chem.CombineMols, [editable_mol, *clean_subs_LB])
    editable_mol_2 = Chem.EditableMol(combo_2)

    c_mol_2 = editable_mol_2.GetMol()
    tags_2 = get_tagged_idx(c_mol_2)
    lb_bb = tags_2['LB_Backbone']
    for sub in tags_2.get('Substituent', []):
        editable_mol_2.AddBond(lb_bb[0], sub, order=Chem.rdchem.BondType.SINGLE)

    return editable_mol_2.GetMol()

#Get the number of substitutable sites for each of the two Lewis centers
def get_valency(smiles):
    mol = Chem.MolFromSmiles(smiles)

    valency_dict = {
        'LA': 0,
        'LB': 0
    }

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            neighbors = atom.GetNeighbors()
            for n in neighbors:
                if n.GetSymbol() == 'B':
                    valency_dict['LA'] += 1

                elif n.GetSymbol() in 'N':
                    valency_dict['LB'] += 1

    return valency_dict

#Select substituent fragments for a given backbone
def fragment_selector(fragments_repo, backbone):
    mono_subs = fragments_repo['mono_subs']
    di_subs = fragments_repo['di_subs']

    val_dict = get_valency(backbone)

    selected_fragments = {
        'LA': None,
        'LB': None
    }

    for key, value in val_dict.items():
        if value < 2:
            selected_fragments[key] = random.choices(mono_subs, k=value)
        elif value == 2:
            sub_type = random.choice(['mono', 'di'])
            if sub_type == 'mono':
                selected_fragments[key] = random.choices(mono_subs, k=2)
            elif sub_type == 'di':
                selected_fragments[key] = random.choices(di_subs, k=1)

    return selected_fragments


if __name__ == '__main__':

    random.seed(42)

    csv_file = './FLP_fragments_list_1.csv'
    df = pd.read_csv(csv_file)

    clean_dict = {col_name:df[col_name].dropna().to_list() for col_name in df.columns}

    assembled_smiles = []

    for backbone in df['backbone']:
        try:
            frags_dict = fragment_selector(clean_dict, backbone)
            mol_obj = assembler(backbone, frags_dict)
            assembled_smiles.append(Chem.MolToSmiles(mol_obj))
        except Exception as e:
            print(e)

    save_df = pd.DataFrame({'SMILES': assembled_smiles, 'Backbones': df['backbone'].to_list()})
    save_df.to_csv('./assembled_smiles_1.csv')
