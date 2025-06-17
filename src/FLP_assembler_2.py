import random
from typing import List, Tuple
import rdkit
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdmolops import ReplaceSubstructs
from functools import reduce
import pandas as pd


def choose_lewis_type(bb_mol: Mol, lewis_types=None) -> List[str] | None:

    """
    Assigns Lewis acid or base to the sites on the backbone
    Args:
        bb_mol: Backbone as the mol object
        lewis_types: List containing the atom types of Lewis acids and bases in the backbone

    Returns: List of Lewis acid/base type to be assigned to dummy atoms on the backbone

    """

    if lewis_types is None:
        lewis_types = ['B', 'N']
    lewis_centers = list(lewis_types)

    #Get indices of Lewis sites
    lc_sites_idx = [atom.GetIdx() for atom in bb_mol.GetAtoms() if atom.GetSymbol() == '*']

    n_sites = len(lc_sites_idx)

    #Assign atom types to Lewis sites
    if n_sites > 0:
        lewis_centers += random.choices(lewis_types, k=n_sites - len(lewis_types))
        random.shuffle(lewis_centers)
        return lewis_centers

    else:
        return None

def create_mono_sub(atom_type: str, substituent: str) -> Tuple[Mol, int]:
    """
    Generates a monosubstituted Lewis center for insertion on backbone
    Args:
        atom_type: Atom type of the Lewis center
        substituent: SMILES string of the substituent fragment to be attached to Lewis center

    Returns:
        substituted_mol: Mol object of molecule with substituent attached to Lewis center
        lewis_idx: Index of Lewis center to facilitate attachment of fragment to backbone

    """
    atom_mol = Chem.MolFromSmiles(atom_type)
    subs_mol = Chem.MolFromSmiles(substituent)
    dummy_mol = Chem.MolFromSmiles('*')

    #Connect substituent to Lewis center
    mol_tuple = ReplaceSubstructs(subs_mol, dummy_mol, atom_mol)

    substituted_mol = mol_tuple[0]

    #Get index of Lewis center
    lewis_idx = [atom.GetIdx() for atom in substituted_mol.GetAtoms() if atom.GetSymbol() == atom_type]

    return substituted_mol, lewis_idx[0]


def create_di_sub(atom_type: str, substituents: list) -> Tuple[Mol, int] | None:
    """
    Generates a disubstituted or ringed Lewis center for insertion on backbone
    Args:
        atom_type: Atom type of the Lewis center
        substituents: List of SMILES string of the substituent fragments to be attached to Lewis center

    Returns:
       edited_mol: Mol object of molecule with substituents attached to Lewis center
       lewis_idx: Index of Lewis center to facilitate attachment of fragment to backbone

    """
    atom_mol = Chem.MolFromSmiles(atom_type)
    subs_mol = [Chem.MolFromSmiles(sub) for sub in substituents]

    combined = reduce(Chem.CombineMols, [atom_mol, *subs_mol])
    editable = Chem.EditableMol(combined)

    connection_idx = []
    dummy_idx = []
    lewis_idx = 0

    c_mol = editable.GetMol()

    if c_mol:
        #Get indexes of dummy atoms and atoms in the substituent to be connected to Lewis center
        for atom in c_mol.GetAtoms():
            if atom.GetSymbol() == '*':
                dummy_idx.append(atom.GetIdx())
                neighbors = atom.GetNeighbors()
                assert len(neighbors) == 1, "Multiple neighbors to dummy atom"
                connection_idx.append(neighbors[0].GetIdx())

        #Make bonds between Lewis center and substituents
        editable.BeginBatchEdit()
        for dummy, idx in zip(dummy_idx, connection_idx):
            editable.AddBond(lewis_idx, idx, rdkit.Chem.rdchem.BondType.SINGLE)
            editable.RemoveAtom(dummy)
        editable.CommitBatchEdit()

        edited_mol = editable.GetMol()
        return edited_mol, lewis_idx
    else:
        return None


def assembler(backbone: str, fragment_repo: dict) -> Mol:
    """
    Accepts a backbone structure and places substituted Lewis centers on it to form an IFLP molecule
    Args:
        backbone: SMILES string of the backbone with dummy atoms at Lewis sites
        fragment_repo: Dictionary containing the sustituent fragments

    Returns:
        bb_mol = RDKit Mol object of the generated FLP structure

    """
    bb_mol = Chem.MolFromSmiles(backbone)
    dummy_mol = Chem.MolFromSmiles('*')

    #Get atom types to be placed at the various Lewis sites
    lewis_centers = choose_lewis_type(bb_mol)

    if not lewis_centers:
        return bb_mol

    #Index of the atom type to choose
    lewis_choose_idx = 0
    for atom in bb_mol.GetAtoms():
        if atom.GetSymbol() != '*':
            continue

        #Calculate the available valence of the Lewis site
        available_valence = 3 - atom.GetExplicitValence()
        lewis_atom = lewis_centers[lewis_choose_idx]

        #Monosubstitutes the site if 1 valence is available
        if available_valence == 1:
            sub = random.choice(fragment_repo['mono_subs'])

            frag, idx = create_mono_sub(lewis_atom, sub)
            bb_mol = ReplaceSubstructs(bb_mol, dummy_mol, frag, replacementConnectionPoint=idx)[0]
            lewis_choose_idx += 1

        #Disubstitutes the site if two valences available (either ring or two substituents)
        elif available_valence == 2:
            sub_type = random.choice(['ring', 'mono'])
            if sub_type == 'ring':
                sub = [random.choice(fragment_repo['di_subs'])]
            else:
                sub = random.choices(fragment_repo['mono_subs'], k=2)

            frag, idx = create_di_sub(lewis_atom, sub)
            bb_mol = ReplaceSubstructs(bb_mol, dummy_mol, frag, replacementConnectionPoint=idx)[0]
            lewis_choose_idx += 1

        #In case no substitution is possible the sites are assigned a Lewis atom type
        else:
            lewis_mol = Chem.MolFromSmiles(lewis_atom)
            bb_mol = ReplaceSubstructs(bb_mol, dummy_mol, lewis_mol)[0]
            lewis_choose_idx += 1

    return bb_mol



if __name__ == "__main__":
    csv_path = './FLP_fragments_list.csv'
    df = pd.read_csv(csv_path)
    seed_list = range(50)

    #Remove NaN cells from the dataframe
    clean_dict = {col_name: df[col_name].dropna().to_list() for col_name in df.columns}

    assembled_flp = {'SMILES': []}

    for s in seed_list:
        random.seed(s)

        backbones = clean_dict['backbone']
        assembled_smiles = []
        for bb in backbones:
            try:
                mol = assembler(bb, clean_dict)
                smi = Chem.MolToSmiles(mol)
                canon_smiles = Chem.CanonSmiles(smi)
                assembled_smiles.append(canon_smiles)
            except Exception as e:
                print(f"Exception {e} in SMILES {bb}")

        rm_duplicates = set(assembled_smiles)
        print(f"Total assembled SMILES: {len(rm_duplicates)} for seed: {s}")
        assembled_flp['SMILES'].extend(list(rm_duplicates))

    save_csv = pd.DataFrame(assembled_flp)
    save_csv.to_csv('./assembled_FLP.csv')