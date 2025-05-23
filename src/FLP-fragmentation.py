from rdkit import Chem
from rdkit.Chem import rdmolops
import networkx as nx
from rdkit.Chem.Draw import MolsToGridImage
import matplotlib.pyplot as plt
import numpy as np

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
        mol = Chem.MolFromSmiles(frag)
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        if 'B' and LB_type in atom_symbols:
            if asMol:
                backbone.append(Chem.MolFromSmiles(frag))
            else:
                backbone.append(frag)
        else:
            if asMol:
                substituents.append(Chem.MolFromSmiles(frag))
            else:
                substituents.append(frag)

    return backbone, substituents

def fragment_FLP(flp_mol):
    LA_idx, LB_idx = get_lewis_centers(flp_mol)
    graph = mol_to_nx(flp_mol)

    backbone_neighbor_idx = get_backbone_neighbors(graph, LA_idx, LB_idx)

    bond_dicts = {}
    for l_idx in [LA_idx, LB_idx]:
        bond_dicts.update(get_bonds_to_break(flp_mol, graph, backbone_neighbor_idx, l_idx))

    mol_frags = Chem.FragmentOnBonds(flp_mol, bondIndices=list(bond_dicts.keys()), bondTypes=list(bond_dicts.values()))

    backbone, substituents = gather_backbones_substituents(mol_frags, asMol=True)

    return backbone, substituents

bb, subs = fragment_FLP(flp_mol=test_mol)

def remove_labels(smiles):
    #og_mol = Chem.MolFromSmiles(smiles)
    search_patt = Chem.MolFromSmiles('*')
    clean_mol = rdmolops.ReplaceSubstructs(smiles, search_patt, search_patt, replaceAll=True)
    clean_smile = Chem.MolToSmiles(clean_mol[0])

    return clean_smile

for b in bb:
    print(remove_labels(b))

for sub in subs:
    print(remove_labels(sub))