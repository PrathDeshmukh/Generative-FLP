import os
import glob
from rdkit import Chem
from rdkit.Chem import rdmolops
import networkx as nx
import pickle
import pandas as pd

test_smiles = 'CC(C)c1cccc(C(C)C)c1N1B(c2ccccc2)c2ccccc2C1C'
test_smiles2 = 'Cc1ccc(cc1)S(=O)(=O)N(Cc1ccccc1)C(Cc1ccccc1)B1OC(C)(C)C(C)(C)O1'
test_smiles3 = 'OB1c2ccccc2Nc2ccccc12'
test_smiles4 = 'B(C)C1=CC=CC2=C1C=CC=N2'
test_mol = Chem.MolFromSmiles(test_smiles2)

#Get the Lewis centers in the molecule
def get_lewis_centers(mol, flp_base_type='N'):
    lewis_ctr_dict = {atom.GetSymbol(): atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in ['B', 'N', 'P']}
    LA_idx = lewis_ctr_dict['B']
    LB_idx = lewis_ctr_dict[flp_base_type]

    return LA_idx, LB_idx

# Get all the neighbours of Lewis centers that will be in the FLP backbone
def get_backbone_neighbors(graph, b_idx, n_idx):
    paths_LA_LB = [path for path in nx.all_simple_paths(graph, b_idx, n_idx)]
    backbone_neighbor_idx = list({path[1] for path in paths_LA_LB} | {path[-2] for path in paths_LA_LB})
    return backbone_neighbor_idx

#Get indexes of all the bonds that need to be broken
def get_bonds_to_break(mol, graph, backbone_neighbor_idx, lewis_idx):
    neighbours = set(graph.neighbors(lewis_idx))
    substituents_idx = list(neighbours.difference(backbone_neighbor_idx))
    bond_objs = [Chem.rdchem.Mol.GetBondBetweenAtoms(mol, lewis_idx, sub) for sub in substituents_idx]
    bonds_dict = {b.GetIdx():b.GetBondType() for b in bond_objs}

    return bonds_dict

#Function for generating graph from mol object
def mol_to_nx(mol):
    G = nx.Graph()

    G.add_nodes_from((atom.GetIdx(),{
        'atomic_num': atom.GetAtomicNum(),
        'atom_index': atom.GetIdx()
    }) for atom in mol.GetAtoms())

    G.add_edges_from((bond.GetBeginAtomIdx(),
                      bond.GetEndAtomIdx(),
                      {'bond_type': bond.GetBondType()})
                     for bond in mol.GetBonds())
    return G

def gather_backbones_substituents(frag_mol, LB_type='N', asMol=True):
    smiles_frags = Chem.MolToSmiles(frag_mol)
    fragments = smiles_frags.split('.')

    backbone = []
    substituents = []

    for frag in fragments:
        mol = Chem.MolFromSmiles(frag, sanitize=False)

        for atom in mol.GetAtoms():
            if (not atom.IsInRing() and atom.GetIsAromatic()):
                atom.SetIsAromatic(False)


        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        if 'B' and LB_type in atom_symbols:
            if asMol:
                backbone.append(mol)
            else:
                backbone.append(Chem.MolToSmiles(mol))
        else:
            if asMol:
                substituents.append(mol)
            else:
                substituents.append(Chem.MolToSmiles(mol))

    return backbone, substituents

def fragment_FLP(flp_mol):
    LA_idx, LB_idx = get_lewis_centers(flp_mol)
    graph = mol_to_nx(flp_mol)

    backbone_neighbor_idx = get_backbone_neighbors(graph, LA_idx, LB_idx)

    bond_dicts = {}
    for l_idx in [LA_idx, LB_idx]:
        bonds_to_break = get_bonds_to_break(flp_mol, graph, backbone_neighbor_idx, l_idx)
        bond_dicts.update(bonds_to_break)

    if bond_dicts:
        mol_frags = Chem.FragmentOnBonds(flp_mol, bondIndices=list(bond_dicts.keys()), bondTypes=list(bond_dicts.values()))

        backbone, substituents = gather_backbones_substituents(mol_frags, asMol=True)

        return backbone, substituents

    else:
        return [flp_mol], []

def remove_labels(mol):
    search_patt = Chem.MolFromSmiles('*')
    clean_mol = rdmolops.ReplaceSubstructs(mol, search_patt, search_patt, replaceAll=True)
    clean_smile = Chem.MolToSmiles(clean_mol[0])

    return clean_smile

if __name__=='__main__':
    smiles_path = './one_BN_pair_FLP.csv'
    df = pd.read_csv(smiles_path)

    fragments_save = {
        'backbone': [],
        'mono_subs': [],
        'di_subs':[]
    }

    for smi in df['SMILES']:
        try:
            mol = Chem.MolFromSmiles(smi)
            mol_H = Chem.AddHs(mol, explicitOnly=False)
            bb, subs = fragment_FLP(mol_H)

            bb_clean = [remove_labels(b) for b in bb]
            fragments_save['backbone'].extend(bb_clean)

            if len(subs) > 0:
                subs_clean = [remove_labels(s) for s in subs]

                for sub_clean in subs_clean:
                    if sub_clean.count('*') == 1:
                        fragments_save['mono_subs'].append(sub_clean)

                    elif sub_clean.count('*') == 2:
                        fragments_save['di_subs'].append(sub_clean)

        except Exception as e:
            print(f"Exception {e} in SMILES: {smi}")

    for k in fragments_save:
        fragments_save[k] = list(set(fragments_save[k]))

    max_len = max(len(lst) for lst in fragments_save.values())

    for key in fragments_save:
        fragments_save[key] += [None] * (max_len - len(fragments_save[key]))

    save_df = pd.DataFrame(fragments_save)
    save_df.to_csv('./FLP_fragments_list_1.csv')