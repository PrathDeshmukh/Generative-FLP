import ccdc.search
from ccdc import io
from ccdc.molecule import Molecule
import pickle
import pandas as pd

#csd = io.MoleculeReader('CSD')
#mol = csd.molecule('MOYLOC')

csv_path = '/home/pratham/Desktop/backbones_seed_CSD.csv'
df = pd.read_csv(csv_path)
for i, rows in df.iterrows():
    code = rows['Code']
    smarts = rows['SMARTS']

    subs_smarts = ccdc.search.SMARTSSubstructure(smarts)
    search = ccdc.search.SubstructureSearch()
    search.add_substructure(subs_smarts)
    hits = search.search()

    print(f"No. of hits for substructure {code}: {len(hits)}")

    rejected_molecules = []
    accepted_molecules = []
    for hit in hits:
        molecule = hit.molecule

        # Check disjoint molecules
        components = molecule.components
        temp_accepted = []
        for frag in components:
            at_symbols = [atom.atomic_symbol for atom in frag.atoms]
            if all(symbol in at_symbols for symbol in ('B', 'N')):
                temp_accepted.append(frag)

        for mol in temp_accepted:
            smiles = mol.smiles
            #Check for metal in the molecule
            if any(atom.is_metal for atom in mol.atoms):
                rejected_molecules.append(smiles)
                continue

            #Check if the molecule has charged atoms
            if mol.has_charged_atoms:
                rejected_molecules.append(smiles)
                continue

            #Check for multiple Boron atoms
            B_count = sum(1 for atom in mol.atoms if atom.atomic_symbol == 'B')
            if B_count > 2:
                rejected_molecules.append(smiles)
                continue

            #Check if Boron accepts a lone pair
            too_many_neighbours = False
            for atom in mol.atoms:
                if atom.atomic_symbol in ('B', 'N') and len(atom.neighbours) > 3:
                    too_many_neighbours = True
                    break

            if too_many_neighbours:
                rejected_molecules.append(smiles)
                continue

            accepted_molecules.append(smiles)

    print(f"Accepted molecules: {len(accepted_molecules)}")
    print(f"Rejected molecules: {len(rejected_molecules)}")

    with open(f"./CSD_backbones/CSD_{code}.pkl", 'wb') as f:
        pickle.dump({'SMILES': accepted_molecules}, f)
        f.close()
    print("Saved SMILES in pickle file")