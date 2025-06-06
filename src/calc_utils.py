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

import heapq

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

    calc_opt = Gaussian(mem='8GB',
                        nprocshared=24,
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

def get_confs_ff(mol, maxiters=250):
    mol_copy = Chem.Mol(mol)
    mol_structure = Chem.Mol(mol)
    mol_structure.RemoveAllConformers()
    try:
        if Chem.rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFSanitizeMolecule(mol)
            energies = AllChem.MMFFOptimizeMoleculeConfs(
                mol, maxIters=maxiters, nonBondedThresh=15.0
            )
            energies_list = [e[1] for e in energies]
            min_e_index = energies_list.index(min(energies_list))
            mol_structure.AddConformer(mol.GetConformer(min_e_index))
            return mol_structure
        else:
            logger.debug(
                "Could not do complete MMFF typing. SMILES {0}".format(mol2smi(mol))
            )
    except ValueError:
        logger.debug(
            "Conformational sampling led to crash. SMILES {0}".format(mol2smi(mol))
        )
        mol = mol_copy
    try:
        if Chem.rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
            energies = AllChem.UFFOptimizeMoleculeConfs(
                mol, maxIters=maxiters, vdwThresh=15.0
            )
            energies_list = [e[1] for e in energies]
            min_e_index = energies_list.index(min(energies_list))
            mol_structure.AddConformer(mol.GetConformer(min_e_index))
            return mol_structure
        else:
            logger.debug(
                "Could not do complete UFF typing. SMILES {0}".format(mol2smi(mol))
            )
    except ValueError:
        logger.debug(
            "Conformational sampling led to crash. SMILES {0}".format(mol2smi(mol))
        )
        mol = mol_copy
    logger.debug(
        "Conformational sampling not performed. SMILES {0}".format(mol2smi(mol))
    )
    return mol_copy

def get_structure_ff(mol, n_confs=5):
    """Generates a reasonable set of 3D structures
    using forcefields for a given rdkit.mol object.
    It will try several 3D generation approaches in rdkit.
    It will try to sample several conformations and get the minima.

    Parameters:
    :param mol: an rdkit mol object
    :type mol: rdkit.mol
    :param n_confs: number of conformations to sample
    :type n_confs: int

    Returns:
    :return mol_structure: mol with 3D coordinate information set
    """
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    coordinates_added = False
    if not coordinates_added:
        try:
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=n_confs,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                pruneRmsThresh=1.25,
                enforceChirality=True,
            )
        except BaseException:
            logger.debug("Method 1 failed to generate conformations.")
        else:
            if all([conformer_id >= 0 for conformer_id in conformer_ids]):
                coordinates_added = True

    if not coordinates_added:
        try:
            params = params = AllChem.srETKDGv3()
            params.useSmallRingTorsions = True
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol, numConfs=n_confs, params=params
            )
        except BaseException:
            logger.debug("Method 2 failed to generate conformations.")
        else:
            if all([conformer_id >= 0 for conformer_id in conformer_ids]):
                coordinates_added = True

    if not coordinates_added:
        try:
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=n_confs,
                useRandomCoords=True,
                useBasicKnowledge=True,
                maxAttempts=250,
                pruneRmsThresh=1.25,
                ignoreSmoothingFailures=True,
            )
        except BaseException:
            logger.debug("Method 3 failed to generate conformations.")
        else:
            if all([conformer_id >= 0 for conformer_id in conformer_ids]):
                coordinates_added = True
        finally:
            if not coordinates_added:
                diagnose_mol(mol)

    if not coordinates_added:
        logger.exception(
            "Could not embed the molecule. SMILES {0}".format(mol2smi(mol))
        )
        return mol
    else:
        mol_structure = get_confs_ff(mol, maxiters=250)
        return mol_structure

def cleanup_mol(mol: Chem.Mol, n_confs: int = 20) -> Chem.Mol:
    """
    Clean up a molecule with potentially bad conformer information.

    Parameters:
    -----------
    mol : Chem.Mol
        The molecule with potentially bad conformer information.

    n_confs : int, optional
        The desired number of conformers to generate (default is 20).

    Returns:
    --------
    Chem.Mol
        A molecule object with cleaned conformer information.
    """

    try:
        # Attempt to generate conformers using 'get_structure_ff'
        mol = get_structure_ff(mol, n_confs)

    except Exception as e:
        print(e)
        # If 'get_structure_ff' fails, generate a single random conformer
        conf_id = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=1,
            useRandomCoords=True,
            numThreads=multiprocessing.cpu_count(),
        )

    return mol

def xtb_opt(xyz, charge=0, unpaired_e=0, level=1):
    """
    Perform quick and reliable geometry optimization with xtb module
    Parameters:
    1. xyz: xyz file
    2. charge: (int)
    3. unpaired_e: number of unpaired electrons
    4. level: 0-2; Halmitonian level [gfn-1,2, gfnff], default=2

    Returns:
    none
    (filename_xtbopt.xyz)
    """
    execution = []
    if level == 0:
        execution = ["xtb", "--gfnff", xyz, "--opt"]
    elif level == 1:
        execution = [
            "xtb",
            "--gfn",
            "1",
            xyz,
            "--opt",
        ]
    elif level == 2:
        execution = [
            "xtb",
            "--gfn",
            "2",
            xyz,
            "--opt",
        ]

    if charge != 0:
        execution.extend(["--charge", str(charge)])
    if unpaired_e != 0:
        execution.extend(["--uhf", str(unpaired_e)])
    sp.call(execution, stdout=sp.DEVNULL, stderr=sp.STDOUT)

    name = xyz[:-4]

    try:
        os.rename("xtbopt.xyz", f"{name}_xtbopt.xyz")
    except Exception as e:
        os.rename("xtblast.xyz", f"{name}_xtbopt.xyz")


def xtb_opt_(xyz, save_folder, charge: int):
    filename = os.path.basename(xyz)
    atoms = read(xyz)
    atoms.calc = XTB(method="GFN2-xTB", charge=charge)

    opt_xtb = BFGS(atoms)
    opt_xtb.run()

    save_path = os.path.join(save_folder, f"/xtb_optimized/{filename}")
    atoms.write(save_path)
    return save_path

if __name__ == '__main__':
    xyz1 = '/home/ppdeshmu/test_xyz.xyz'
    test_e = get_dft_E(xyz=xyz1, label='0')
    print(test_e)