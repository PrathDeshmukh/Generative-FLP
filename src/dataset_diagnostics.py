import pandas as pd
import numpy as np
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit.Chem.rdmolfiles import MolFromSmiles

import matplotlib.pyplot as plt

#Molecule properties to be calculated no. of heavy atoms, heter atoms, aromatic rings, aliphatic rings, CalcFractionCSP3, VDWvolume

path = '/home/pratham/PycharmProjects/ga_flp_ruben/Literature-SMILES.csv'

df = pd.read_csv(path)
smiles = np.array(df['SMILES'].values)

properties = {
    'nha' :rdmd.CalcNumHeavyAtoms,
    'nhta':rdmd.CalcNumHeteroatoms,
    'nar' :rdmd.CalcNumAromaticRings,
    'nalr':rdmd.CalcNumAliphaticRings,
    'fsp3':rdmd.CalcFractionCSP3
    #'vdwv':[]
}

prop_dict = {key: [] for key in properties.keys()}

def calc_props(smi: str):
    try:
        mol = MolFromSmiles(smi)
        if mol is not None:
            mol.UpdatePropertyCache()
            return {prop: func(mol) for prop, func in properties.items()}
    except Exception as e:
        print(f"Exception {e} in SMILES: {smi}")

for smi in smiles:
    props = calc_props(smi)
    print(props)
    if props:
        for prop, val in props.items():
            prop_dict[prop].append(val)


fig, axs = plt.subplots(3,2)
axes = axs.flatten()

for prop, ax in zip(prop_dict.keys(), axes):
    ax.hist(prop_dict[prop])
    ax.set_title(f'{prop}')

plt.show()