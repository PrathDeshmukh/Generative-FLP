import pandas as pd
from PIL import Image
from io import BytesIO
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage

def add_subs_markers(frag_type: str, smiles: str) -> str:
    if "*" in smiles and frag_type=='BB':
        smiles = smiles[1:]+'N1'

    elif "*" in smiles and frag_type=='LBr':
        smiles = smiles.replace('*', 'N1')

    elif "*" in smiles and frag_type=='LAr':
        smiles = smiles.replace('*', 'B1')

    if "()" in smiles:
        smiles = smiles.replace("()", "([H])")

    if "@" in smiles and "$" in smiles:
        smiles = smiles[2:]+'n1'

    if "@" in smiles:
        smiles = smiles[1:]+'N12'

    return smiles

if __name__=='__main__':
    path_csv = "/home/ppdeshmu/Generative-FLP/data/database_HC_bbr.csv"

    df = pd.read_csv(path_csv)
    column_names = list(df)

    smiles_dict = {cname: list(df[cname].dropna()) for cname in column_names}

    for frag_type, smiles_list in smiles_dict.items():
        mol_list = [Chem.MolFromSmiles(add_subs_markers(frag_type=frag_type, smiles=smiles)) for smiles in smiles_list]
        imgs = MolsToGridImage(mol_list, useSVG=False, returnPNG=True)
        image = Image.open(BytesIO(imgs))
        image.save(f'/home/ppdeshmu/Generative-FLP/data/{frag_type}-structures.pdf')
        print(f"{frag_type} structures saved")