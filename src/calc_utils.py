import logging
import os
from ase.io import read
from ase.calculators.gaussian import Gaussian
from xtb.ase.calculator import XTB
from ase.optimize import BFGS
import numpy as np
import multiprocessing
import subprocess as sp
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToSmiles as mol2smi


logger = logging.getLogger(__name__)
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog("rdApp.*")
logging.getLogger("rdkit").setLevel(logging.CRITICAL)


n_cores_ = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = n_cores_
os.environ["OPENBLAS_NUM_THREADS"] = n_cores_
os.environ["MKL_NUM_THREADS"] = n_cores_
os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores_
os.environ["NUMEXPR_NUM_THREADS"] = n_cores_

def get_dft_E(xyz, label, charge:int = 0):
    atoms = read(xyz)

    label_type = None
    if charge == 0:
        label_type = "INIT"
    elif charge == 1:
        label_type = "FEPA"
    elif charge == -1:
        label_type = "FEHA"

    calc_opt = Gaussian(mem='28GB',
                        nprocshared=12,
                        label=os.path.join('/home/ppdeshmu/scratch/', f"{label_type}_{label}"),
                        xc='PBE1PBE',
                        basis='def2TZVP',
                        command='g16 < PREFIX.com> PREFIX.log',
                        charge=charge,
                        extra='EmpiricalDispersion=GD3BJ'
                        )

    atoms.calc = calc_opt
    energy = atoms.get_potential_energy()

    return energy

def unit_vector(v):
    return v / np.linalg.norm(v)

def angle_between(u, v):
    v_1 = unit_vector(v)
    u_1 = unit_vector(u)
    return np.arccos(np.clip(np.dot(v_1, u_1), -1.0, 1.0))


def xtb_opt_(xyz, save_folder, charge: int):
    filename = os.path.basename(xyz)
    atoms = read(xyz)
    atoms.calc = XTB(method="GFN2-xTB", charge=charge)

    opt_xtb = BFGS(atoms)
    opt_xtb.run()

    save_path = os.path.join(save_folder, f"xtb_optimized/{filename}")
    atoms.write(save_path)
    return save_path

def get_lewis_h_dist(xyz, lewis_idx):
    atoms = read(xyz)
    pos = atoms.get_positions()
    H_pos = pos[-1]
    lewis_pos = pos[lewis_idx]

    dist = np.linalg.norm(lewis_pos - H_pos)
    return dist

if __name__ == '__main__':
    xyz1 = '/home/ppdeshmu/test_xyz.xyz'
    test_e = get_dft_E(xyz=xyz1, label='0')
    print(test_e)