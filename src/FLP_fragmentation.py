from rdkit import Chem
from rdkit.Chem import rdmolops
import networkx as nx
from networkx.classes.graph import Graph
from rdkit.Chem import Mol
import pandas as pd
from typing import List, Tuple, Union


def get_lewis_centers(mol: Mol, flp_base_type: str = 'N') -> Tuple[List[int], List[int]]:
    """
    Get the Lewis centers in the molecule
    Args:
        mol: RDKit Mol object of the FLP molecule
        flp_base_type: Atom type of Lewis base present in the molecule

    Returns:
        LA_idx, LB_idx: Indexes of Lewis acid and base centers
    """
    lewis_ctr_dict = {atom_symbol: [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == atom_symbol] for atom_symbol in ['B', 'N', 'P']}
    LA_idx = lewis_ctr_dict['B']
    LB_idx = lewis_ctr_dict[flp_base_type]

    return LA_idx, LB_idx


def get_backbone_neighbors(graph: Graph, b_idx: List[int], n_idx: List[int]) -> List[int]:
    """
    Get all the neighbours of Lewis centers which will subsequently be present in the FLP backbone
    Args:
        graph: Networkx graph object of the molecule
        b_idx: List of indices of Lewis acid site
        n_idx: List of indices of Lewis base site

    Returns:
        backbone_neighbor_idx: List of indices of backbone neighbor atoms

    """
    #Get all the paths going from B to N
    paths_LA_LB = [path for b in b_idx for n in n_idx for path in nx.all_simple_paths(graph, b, n) ]

    #Get indices of atoms that are neighbors of the Lewis centers and that are in the backbone
    backbone_neighbor_idx = list({path[1] for path in paths_LA_LB} | {path[-2] for path in paths_LA_LB})

    return backbone_neighbor_idx


def get_bonds_to_break(mol: Mol, graph: Graph, backbone_neighbor_idx: List[int], lewis_idx: List[int]) -> dict:
    """
    Get indexes of all the bonds that need to be broken
    Args:
        mol: RDKit Mol object of IFLP molecule
        graph: Networkx graph object of IFLP molecule
        backbone_neighbor_idx: Indices of backbone neighbor atoms
        lewis_idx: Indices of Lewis centers

    Returns:
        bonds_dict: Dictionary containing the bonds to be broken
    """
    #Get all neighbors of Lewis sites
    neighbours = set()
    for l_idx in lewis_idx:
        neighbours.update(graph.neighbors(l_idx))

    #Get indices of neighbors that are part of substituent that will subsequently be removed
    substituents_idx = list(neighbours.difference(backbone_neighbor_idx))

    #Get indices of bonds to substituents that will be broken
    bond_objs = []
    for l in lewis_idx:
        for sub in substituents_idx:
            bond = mol.GetBondBetweenAtoms(l, sub)
            if bond is not None:
                bond_objs.append(bond)

    bonds_dict = {b.GetIdx():b.GetBondType() for b in bond_objs}

    return bonds_dict


def mol_to_nx(mol: Mol) -> Graph:
    """
    Function for generating networkx graph from RDKit mol object
    Args:
        mol: RDKit mol object of IFLP molecule

    Returns:
        G: Networkx graph object of IFLP molecule
    """
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

def gather_backbones_substituents(frag_mol: Mol, LB_type='n') -> Tuple[List[str], List[str]]:
    """
    Separates backbone and substituents in the fragmented molecule
    Args:
        frag_mol: RDKit mol object of fragmented molecule
        LB_type: Lewis base atom type

    Returns:
        backbone, substituents: Lists containing SMILES of backbone and substituents respectively
    """
    fragments = Chem.MolToSmiles(frag_mol).split('.')

    backbone, substituents = [], []

    for frag in fragments:
        smiles_lower = frag.lower()
        if 'b' in smiles_lower and LB_type in smiles_lower:
            backbone.append(frag)
        else:
            substituents.append(frag)

    return backbone, substituents

def clean_backbone(backbone: Union[str, Mol]) -> Mol:
    """
    Replaces all the Lewis acid and base atoms on the backbone with dummy atoms
    Args:
        backbone: Backbone with substituents removed

    Returns:
        clean_mol: RDKit mol object of the cleaned backbone

    """
    if isinstance(backbone, str):
        bb_smiles = backbone
        bb_mol = Chem.MolFromSmiles(bb_smiles, sanitize=False)

    elif isinstance(backbone, Mol):
        bb_mol = backbone

    else:
        raise ValueError("Backbone should be either SMILES or RDKit Mol object!")

    rwmol = Chem.RWMol(bb_mol)

    rwmol.BeginBatchEdit()

    for atom in rwmol.GetAtoms():
        if atom.GetSymbol() == '*':
            rwmol.RemoveAtom(atom.GetIdx())

    rwmol.CommitBatchEdit()

    for atom in rwmol.GetAtoms():
        if atom.GetSymbol() in ['B', 'N']:
            atom.SetAtomicNum(0)


    clean_mol = rwmol.GetMol()
    return clean_mol


def fragment_FLP(flp_mol: Mol) -> Tuple[List[Mol], List[str]]:
    """
    Main fragmentation function, accepts FLP molecule, removes substituents and replaces Lewis sites with dummy atoms
    Args:
        flp_mol: RDKit Mol object of FLP molecule

    Returns:
        backbone: RDKit mol object of the backbone of FLP
        substituents: List of SMILES string of substituents
        flp_mol_H: Returns FLP as it is if no substituents available

    """
    flp_mol_H = Chem.AddHs(flp_mol, explicitOnly=False)
    LA_idx, LB_idx = get_lewis_centers(flp_mol_H)
    graph = mol_to_nx(flp_mol_H)
    backbone_neighbor_idx = get_backbone_neighbors(graph, LA_idx, LB_idx)


    bond_dicts = {}
    for l_idx in [LA_idx, LB_idx]:
        bonds_to_break = get_bonds_to_break(flp_mol_H, graph, backbone_neighbor_idx, l_idx)
        bond_dicts.update(bonds_to_break)

    # Break bonds to substituents
    if bond_dicts:
        mol_frags = Chem.FragmentOnBonds(flp_mol_H, bondIndices=list(bond_dicts.keys()), bondTypes=list(bond_dicts.values()))
        backbone, substituents = gather_backbones_substituents(mol_frags)
        backbone = [clean_backbone(backbone[0])]
        return backbone, substituents

    else:
        return [clean_backbone(flp_mol_H)], []


if __name__=='__main__':

    smiles_path = './one_B_three_N.csv'
    df = pd.read_csv(smiles_path)

    fragments_save = {
        'backbone': [],
        'mono_subs': [],
        'di_subs':[]
    }

    for smi in df['SMILES']:
        try:
            mol = Chem.MolFromSmiles(smi)
            bb, subs = fragment_FLP(mol)

            bb_smiles = [Chem.MolToSmiles(b) for b in bb]
            fragments_save['backbone'].extend(bb_smiles)

            if len(subs) > 0:
                for sub_clean in subs:
                    if sub_clean.count('*') == 1:
                        fragments_save['mono_subs'].append(sub_clean)

                    elif sub_clean.count('*') == 2:
                        fragments_save['di_subs'].append(sub_clean)

        except Exception as e:
            print(f"Exception {e} in SMILES: {smi}")

    for k in fragments_save:
        frag_list = list(set(fragments_save[k]))
        fragments_save[k] = frag_list
        print(f'{k}: {len(frag_list)}')

    max_len = max(len(lst) for lst in fragments_save.values())

    for key in fragments_save:
        fragments_save[key] += [None] * (max_len - len(fragments_save[key]))

    save_df = pd.DataFrame(fragments_save)
    save_df.to_csv('./FLP_fragments_list.csv')