from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdchem
from rdkit.Chem.Draw import MolsToGridImage
from functools import reduce

substituents = ['*Cc1ccccc1','*S(=O)(=O)c1ccc(C)cc1','*OC(C)(C)C(C)(C)O*']
backbones = ['*B(*)C(Cc1ccccc1)N(*)*']

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
        if atom.GetIdx() in idx and atom.GetSymbol() in ['B', 'N', 'P']:
            atom.SetProp('ConnectType', 'Backbone')

        elif atom.GetIdx() in idx:
            atom.SetProp('ConnectType', 'Substituent')

        else:
            atom.SetProp('ConnectType', 'noConnect')
    return mol


def remove_dummy_atoms(mol):
    dummy_idx = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            dummy_idx.append(atom.GetIdx())

    rwmol = rdchem.RWMol(mol)
    for d_idx in reversed(dummy_idx):
        rwmol.RemoveAtom(d_idx)

    return rwmol.GetMol()

def get_tagged_idx(mol):
    idx_dict = {
        'backbone': [],
        'substituent': []
    }

    for atom in mol.GetAtoms():
        if atom.GetProp('ConnectType') == 'Backbone':
            idx_dict['backbone'].append(atom.GetIdx())

        elif atom.GetProp('ConnectType') == 'Substituent':
            idx_dict['substituent'].append(atom.GetIdx())

    return idx_dict


clean_bb = remove_dummy_atoms(set_connectivity_tags(backbones[0]))
clean_sub = remove_dummy_atoms( set_connectivity_tags(substituents[0]))
clean_sub2 = remove_dummy_atoms(set_connectivity_tags(substituents[0]))

combo = reduce(Chem.CombineMols, [clean_bb, clean_sub, clean_sub2])
combo_editable = Chem.EditableMol(combo)

c_mol = combo_editable.GetMol()

bb1 = get_tagged_idx(c_mol)['backbone']
sub1 = get_tagged_idx(c_mol)['substituent']

combo_editable.AddBond(bb1[0], sub1[0], order=Chem.rdchem.BondType.SINGLE)
combo_editable.AddBond(bb1[0], sub1[1], order=Chem.rdchem.BondType.SINGLE)

combined_mol = combo_editable.GetMol()
print(Chem.MolToSmiles(combined_mol))