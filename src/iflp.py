import csv
import logging
import os
from typing import Tuple, List
import pandas as pd
import numpy as np
import submitit

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles

from ase.io import read
from calc_utils import get_shortest_BN_distance, get_H2_pos, get_dft_E, xtb_opt_, get_lewis_h_dist, angle_between, calc_directed_vector


class IFLP:
    def __init__(
        self,
        threshold: float = 1.8,
        level: int = 2,
        verb: int = 0,
        label: int = 0,
        save_folder: str="/home/ppdeshmu/Generative-FLP/data/xyz"
    ):
        self.threshold = threshold
        self.level = level
        self.verb = verb
        self.label = label
        self.save_folder = save_folder
        self.mol_relaxed_xyz = None
        self.idx_dict = None

        os.makedirs(os.path.join(save_folder, "initial_FLP"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "xtb_optimized"), exist_ok=True)


    def calc_FEPA_FEHA(self, mol_FLP: Chem.Mol) -> Tuple[float, float]:
        """
        Calculate the Free Energy of Proton Affinity (FEPA) and Free Energy of Hydride Affinity (FEHA)
        for a given relaxed FLP molecule.

        Parameters:
        - mol_FLP (Chem.Mol): The FLP molecule for which FEPA and FEHA will be calculated.

        Returns:
        Tuple[float, float]: A tuple containing FEPA and FEHA values in kcal/mol.
        If the calculation fails, (1e10, 1e10) is returned for both.
        """

        E_PROTON = 0.00
        E_HYDRIDE = -0.527751
        h2kcal = 627.509

        mol_FLP = Chem.AddHs(mol_FLP)
        AllChem.EmbedMolecule(mol_FLP)

        conf = mol_FLP.GetConformer()
        coords = conf.GetPositions()

        cm = Chem.rdmolops.GetAdjacencyMatrix(mol_FLP)

        _, B_idx, N_idx = get_shortest_BN_distance(mol_FLP, coords, self.threshold)
        z = [atom.GetAtomicNum() for atom in mol_FLP.GetAtoms()]

        H_a, H_b, subs_acid_idx, subs_base_idx = get_H2_pos(z, B_idx, N_idx, cm, coords, self.verb)
        if (H_a is None) or (H_b is None):
            if self.verb > 0:
                logging.error(
                    f"Fail to generate H positions for {Chem.MolToSmiles(mol_FLP)}"
                )
            raise ValueError("Fail to generate H positions")
        else:
            self.idx_dict = {
                'B_idx': B_idx,
                'N_idx': N_idx,
                'subs_acid_idx': subs_acid_idx,
                'subs_base_idx': subs_base_idx
            }

            init_xyz_path = os.path.join(self.save_folder, f"initial_FLP/{self.label}.xyz")
            rdmolfiles.MolToXYZFile(mol_FLP, init_xyz_path)

            opt_flp_path = xtb_opt_(init_xyz_path, self.save_folder, charge=0)

            ha_pos = f"H {H_a[0]:13.6f}{H_a[1]:12.6f}{H_a[2]:12.6f}\n"
            hb_pos = f"H {H_b[0]:13.6f}{H_b[1]:12.6f}{H_b[2]:12.6f}"

            with open(init_xyz_path, "r") as f:
                contents = f.readlines()

            contents_b = contents.copy()
            contents_c = contents.copy()

            #Generate and relax intermediate 2 of FLP
            contents_c.append(ha_pos)
            contents_c.append(hb_pos)

            flp_h2_path = os.path.join(self.save_folder, f"initial_FLP/FLP_H2_{self.label}.xyz")
            with open(flp_h2_path, "w") as f:
                contents_c[0] = str(len(contents_c[2:])) + "\n"
                f.writelines(contents_c)

            opt_flp_h2_path = xtb_opt_(flp_h2_path, self.save_folder, charge=0)
            self.mol_relaxed_xyz = opt_flp_h2_path

            # FEHA
            contents.append(ha_pos)
            contents[0] = str(len(contents[2:])) + "\n"

            flp_BH_path = os.path.join(self.save_folder, f"initial_FLP/FLP_BH_{self.label}.xyz")

            with open(flp_BH_path, "w") as f:
                f.writelines(contents)

            opt_BH_path = xtb_opt_(flp_BH_path, self.save_folder, charge=-1)

            d_bh = get_lewis_h_dist(opt_BH_path, B_idx)

            if d_bh > 1.5:
                if self.verb > 0:
                    logging.error(
                        f"Fail to generate relaxed FLP-BH for {Chem.MolToSmiles(mol_FLP)}"
                    )
                raise ValueError("Fail to generate relaxed FLP-BH")

            # FEPA
            contents_b.append(hb_pos)
            contents_b[0] = str(len(contents_b[2:])) + "\n"

            flp_NH_path = os.path.join(self.save_folder, f"initial_FLP/FLP_NH_{self.label}.xyz")

            with open(flp_NH_path, "w") as f:
                f.writelines(contents_b)

            opt_NH_path = xtb_opt_(flp_NH_path, self.save_folder, charge=1)

            d_nh = get_lewis_h_dist(opt_NH_path, N_idx)
            if d_nh > 1.5:
                if self.verb > 0:
                    logging.error(
                        f"Fail to generate relaxed FLP-NH for {Chem.MolToSmiles(mol_FLP)}, return 1e10, 1e10"
                    )
                raise ValueError("Fail to generate relaxed FLP-NH")

            #E_iflp = get_dft_E(opt_flp_path, self.label)

            #E_iflp_b = get_dft_E(opt_BH_path, self.label, charge=-1)
            #feha = (E_iflp_b - E_iflp - E_HYDRIDE) * h2kcal

            #E_iflp_n = get_dft_E(opt_NH_path, self.label, charge=1)
            #fepa = (E_iflp_n - E_iflp - E_PROTON) * h2kcal
            fepa, feha = 0, 0
        return fepa, feha

    def geom_targets(self) -> Tuple[float, float]:
        """
        Calculate geometric properties of a relaxed molecule.

        Parameters:
        - mol_relaxed (Chem.Mol): The relaxed molecule for which geometric properties will be calculated.

        Returns:
        Tuple[float, float]: A tuple containing the best B-N distance and the angle (eta) in degrees.
        If the calculation fails, an appropriate exception is raised.
        """
        try:
            B_pos = self.idx_dict['B_idx']
            N_pos = self.idx_dict['N_idx']
            subs_acid_idx = self.idx_dict['subs_acid_idx']
            subs_base_idx = self.idx_dict['subs_base_idx']

            atoms = read(self.mol_relaxed_xyz)
            coords = atoms.get_positions()

            best_dist = np.linalg.norm(coords[B_pos] - coords[N_pos])

            vec_acid = calc_directed_vector(B_pos, subs_acid_idx, coords)
            vec_base = calc_directed_vector(N_pos, subs_base_idx, coords)

            eta = angle_between(vec_acid, vec_base) * 180 / np.pi
            return best_dist, eta

        except Exception as e:
            raise ValueError(f"Fail to compute geometric properties: {str(e)}") from e

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

if __name__ == "__main__":
    csv_path = "/home/ppdeshmu/Generative-FLP/data/csv/assembled_FLP.csv"
    df = pd.read_csv(csv_path)

    desc_csv = '/home/ppdeshmu/Generative-FLP/data/csv/FLP_dataset.csv'

    header_row = ['label', 'fepa', 'feha', 'd', 'phi']
    with open(desc_csv, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(header_row)
      f.close()

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