import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import tight_layout
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdmd
from utils.utils import calcVolume
#from rdkit.Chem.rdmolfiles import MolFromSmiles
import argparse


#Molecule properties to be calculated no. of heavy atoms, heter atoms, aromatic rings, aliphatic rings, CalcFractionCSP3, VDWvolume

def parse_cmd():
    parser = argparse.ArgumentParser(description="Generate t-SNE plots of various databases")
    parser.add_argument(
        "--file_paths",
        nargs='*')
    return parser.parse_args()

#dictionary of properties to be calculated
properties = {
    'nha' :rdmd.CalcNumHeavyAtoms,
    'nhta':rdmd.CalcNumHeteroatoms,
    'nar' :rdmd.CalcNumAromaticRings,
    'nalr':rdmd.CalcNumAliphaticRings,
    'fsp3':rdmd.CalcFractionCSP3,
    'vdwv':calcVolume
}

#property names for plots
prop_names = {
    'nha' :'Num Heavy atoms',
    'nhta':'Num Hetero atoms',
    'nar' :'Num Aromatic Rings',
    'nalr':'Num Aliphatic Rings',
    'fsp3':'Fraction CSP3',
    'vdwv':'Spatial Volume'
}

#function to calculate all the properties for a given SMILES string
def calc_props(smi: str):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol.UpdatePropertyCache()
            return {prop: func(mol) for prop, func in properties.items()}

    except Exception as e:
        print(f"[!] Error processing SMILES {smi}: {e}")
        return None

def main():
    cmd_args = parse_cmd()
    file_paths = cmd_args.file_paths
    if not file_paths:
        print("[!] no files provided.")
        return

    all_props_dict = {}

    for file in file_paths:
        df = pd.read_csv(file)
        smiles = df['SMILES']
        dataset_name = os.path.basename(file).split('.')[0]

        prop_dict = {key: [] for key in properties.keys()}

        for smi in smiles:
            props = calc_props(smi)
            if props is not None:
                for prop, val in props.items():
                    prop_dict[prop].append(val)

        all_props_dict[f'{dataset_name}'] = prop_dict

    #plotting histograms
    fig, axs = plt.subplots(3,2, figsize=(15,15), tight_layout=True)
    axes = axs.flatten()
    #fig.delaxes(axes[-1]) #remove unused sixth plot

    palette = sns.color_palette("bright")
    label_colors = dict(zip(all_props_dict.keys(), palette))

    for file in all_props_dict.keys():
        for prop, ax in zip(all_props_dict[file].keys(), axes):
            sns.histplot(data=all_props_dict[file],
                         x=all_props_dict[file][prop],
                         ax=ax,
                         stat='density',
                         color=label_colors[file])

            ax.set_title(f'{prop_names[prop]}')

    fig.legend(all_props_dict.keys(),
               loc='upper center',
               ncol=len(all_props_dict),
               frameon=False)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    plt.savefig('Generative-FLP/data/plots/properties_histogram.png')
    print("Plot generated!")

if __name__=='__main__':
    main()
