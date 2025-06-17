#import sys
#sys.path.append('/home/ppdeshmu/Generative-FLP/src')
from iflp import IFLP
import pandas as pd
from rdkit import Chem
import submitit
import csv

csv_path = "/home/ppdeshmu/Generative-FLP/data/csv/assembled_FLP.csv"
df = pd.read_csv(csv_path)


desc_csv = '/home/ppdeshmu/Generative-FLP/data/csv/FLP_dataset.csv'

header_row = ['label', 'fepa', 'feha', 'd', 'phi']
with open(desc_csv, 'w') as f:
  writer = csv.writer(f)
  writer.writerow(header_row)
  f.close()

def compute_desc(smiles, label):
    try:
        mol = Chem.MolFromSmiles(smiles)
        iflp_calc = IFLP(label=label)

        fepa, feha = iflp_calc.calc_FEPA_FEHA(mol)
        d, phi = iflp_calc.geom_targets()

        with open(desc_csv, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([label, fepa, feha, d, phi])
        file.close()

    except Exception as e:
        print(f"Exception {e} in SMILES {smiles}")

smi = df['SMILES'].to_list()
labels = range(len(smi))

executor = submitit.AutoExecutor(folder='/home/ppdeshmu/scratch')

if executor._executor.__class__.__name__ == 'SlurmExecutor':
    executor.update_parameters(
        cpus_per_task = 12,
        tasks_per_node = 1,
        slurm_nodes = 1,
        slurm_time = "24:00:00"
    )

jobs = executor.map_array(compute_desc, smi, labels)