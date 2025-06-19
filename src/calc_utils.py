import logging
import os
from ase.io import read
from ase.calculators.gaussian import Gaussian
from xtb.ase.calculator import XTB
from ase.optimize import BFGS
import numpy as np
import multiprocessing
from rdkit import Chem, RDLogger
from typing import List, Tuple
from scipy.spatial import distance
from skspatial.objects import Plane, Points
import ase
from io import StringIO

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

def atoms_to_xyz_str(atoms: ase.Atoms):
  f = StringIO()
  atoms.write(f, format="xyz")
  return f.getvalue()

def unit_vector(v):
    return v / np.linalg.norm(v)

def angle_between(u, v):
    v_1 = unit_vector(v)
    u_1 = unit_vector(u)
    return np.arccos(np.clip(np.dot(v_1, u_1), -1.0, 1.0))

def calc_directed_vector(lewis_idx, subs_idx, coords):
    subs_coords = [coords[i] for i in subs_idx]
    vec = 0.0
    if len(subs_coords) == 2:
        subs_coords.append(lewis_idx)
        points = Points(subs_coords)
        try:
            centroid = points.centroid()
            vec = np.array(coords[lewis_idx]) - centroid
        except ValueError:
            return 0.0

    elif len(subs_coords) == 3:
        points = Points(subs_coords)
        try:
            centroid = points.centroid()
            vec = np.array(coords[lewis_idx]) - centroid
        except ValueError:
            return 0.0

    return vec

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

def calculate_angle(
    B_idx: int, N_idx: int, cm: np.ndarray, coords: np.ndarray, verb: int = 0
) -> float:
    """
    Calculate the angle between two vectors for a given set of atoms.

    Parameters:
    - B_idx (int): Index of the boron atom.
    - N_idx (int): Index of the nitrogen atom.
    - cm (np.ndarray): Adjacency matrix.
    - coords (np.ndarray): Atomic coordinates.
    - verb (int): Verbosity level.

    Returns:
    float
        The angle in degrees between two vectors, or 0.0 if the calculation cannot be performed.
    """
    subs_base = []
    for n, j in enumerate(cm[N_idx, :]):
        if j == 1:
            pos = coords[n]
            pos = np.around(pos, 4)
            subs_base.append(pos)
    if len(subs_base) == 2:
        if verb > 1:
            print("\nsp2 base. Calculating planes.")
        subs_base.append(coords[N_idx, :])
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            if verb > 0:
                print("Cannot calculate centroid. return 0.0")
            return 0.0
        vec_b = np.array(coords[N_idx, :]) - centroid

    elif len(subs_base) == 3:
        if verb > 1:
            print("\nsp3 base. Calculating planes.")
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            if verb > 0:
                print("Cannot calculate centroid. return 0.0")
            return 0.0
        vec_b = np.array(coords[N_idx, :]) - centroid
    else:
        print(
            f"Base substituents are not 2 or 3 but rather {len(subs_base)} Angle set to 0!"
        )
        return 0.0

    vec_b = unit_vector(vec_b)
    subs_acid = []
    for n, j in enumerate(cm[B_idx, :]):
        if j == 1:
            pos = coords[n]
            pos = np.around(pos, 4)
            subs_acid.append(pos)

    if len(subs_acid) == 3:
        if verb > 1:
            print("\nsp3 acid. Calculating planes.")

        points = Points(subs_acid)
        try:
            centroid = points.centroid()
        except ValueError:
            return 0.0
        vec_a = np.array(coords[B_idx, :]) - centroid
    else:
        if verb > 0:
            print("Acid substituents are not 3!")
        return 0.0

    vec_a = unit_vector(vec_a)

    theta = angle_between(vec_a, vec_b) * 180 / np.pi
    return theta


def get_shortest_BN_distance(
    mol: Chem.Mol, coords: np.ndarray, threshold: float
) -> Tuple[float|None, int|None, int|None]:
    """
    Calculate the shortest B-N distance above a specified threshold.

    Parameters:
    - mol (Chem.Mol): The molecular structure.
    - coords (np.ndarray): The coordinates of all atoms in the molecule.
    - threshold (float): The threshold for considering B-N distances.

    Returns:
    tuple: A tuple containing the following:
        - float or None: The shortest B-N distance greater than the threshold, or None if no such distance is found.
        - int or None: The index of the boron (B) atom corresponding to the shortest distance, or None if not found.
        - int or None: The index of the nitrogen (N) atom corresponding to the shortest distance, or None if not found.
    """
    B_dict = {}
    N_dict = {}

    for atom, c in zip(mol.GetAtoms(), coords):
        if atom.GetAtomicNum() == 5:
            B_dict[atom.GetIdx()] = c
        elif atom.GetAtomicNum() == 7:
            N_dict[atom.GetIdx()] = c

    BN_dist_idx = {}

    for bidx, bpos in B_dict.items():
        for nidx, npos in N_dict.items():
            dist = np.linalg.norm(bpos - npos)
            if dist > threshold:
                BN_dist_idx[dist] = (bidx, nidx)
            else:
                continue

    min_dist = min(BN_dist_idx.keys())
    if min_dist:
        B_idx, N_idx = BN_dist_idx[min_dist]
        return min_dist, B_idx, N_idx
    else:
        return None, None, None


def get_H2_pos(
        z: List[int],
        B_idx: int,
        N_idx: int,
        cm: np.ndarray,
        coords: np.ndarray,
        verb: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Determine cartesian coodinates to which hydride and proton are to be added.

    Vector definition on N: The centroid point of substituent groups - Coordinate of N
    Vector definition on B: Coordinate of H(N) - Coordinate of B

    Set bond distance:
    d(B-H) = 1.2
    d(N-H) = 1.2

    Parameters:
    1. z: vector of atomic numbers
    2. N_idx: atomic index of N
    3. B_idx: atomic index of B
    4. cm: connectivity matrix
    5. coords: coordinates matrix
    6. verb: verbosity level

    Returns:
    H_a, H_b: coordinate of H(B) and H(N), respectively
    subs_acid_idx, subs_base_idx: Indexes of atoms around the respective Lewis centers
    """

    # base vector definition
    subs_base = []
    subs_base_idx = []
    for n, j in enumerate(cm[N_idx, :]):
        if j == 1 and z[n] != 5:
            subs_base.append(np.around(coords[n], 4))
            subs_base_idx.append(n)

    if len(subs_base) == 2:
        if verb > 1:
            print("Detect sp2 Nitrogen\n")
        subs_base.append(coords[N_idx, :])
        points = Points(subs_base)
        try:
            centroid = points.centroid()
            vec_b = np.array(coords[N_idx, :]) - centroid
        except ValueError:
            if verb > 0:
                print("Cannot calculate centroid. return None")
            return None, None, None, None

    elif len(subs_base) == 3:
        if verb > 1:
            print("Detect sp3 nitrogen\n")
        points = Points(subs_base)

        try:
            plane = Plane.best_fit(points)
            n1 = plane.normal
            n2 = -n1
            H_b1 = n1 * 1.2
            H_b2 = n2 * 1.2
            d1 = distance.euclidean(H_b1, coords[B_idx, :])
            d2 = distance.euclidean(H_b2, coords[B_idx, :])
            if d1 < d2:
                vec_b = H_b1
            else:
                vec_b = H_b2


        except ValueError:
            if verb > 0:
                print("Cannot determine the base vector. return None")
            return None, None, None, None

    else:
        print(
            f"Base substituents are not 2 or 3 but rather {len(subs_base)} return None"
        )
        return None, None, None, None

    vec_b = unit_vector(vec_b)
    H_b = vec_b * 1.2 + coords[N_idx, :]

    # Acid vector definition
    subs_acid = []
    subs_acid_idx = []
    for n, j in enumerate(cm[B_idx, :]):
        if j == 1 and z[n] != 7:
            subs_acid.append(np.around(coords[n], 4))
            subs_acid_idx.append(n)

    if len(subs_acid) == 2:
        if verb > 1:
            print("Detect sp2 Boron\n")
        subs_acid.append(coords[B_idx, :])
        points = Points(subs_acid)
        try:
            centroid = points.centroid()
            vec_a = np.array(coords[B_idx, :]) - centroid
        except ValueError:
            if verb > 0:
                print("Cannot calculate centroid. return None")
            return None, None, None, None

    elif len(subs_acid) == 3:
        if verb > 1:
            print("Detect sp3 B center\n")
        points = Points(subs_acid)
        try:
            plane = Plane.best_fit(points)
            n1 = plane.normal
            n2 = -n1
            H_a1 = n1 * 1.2
            H_a2 = n2 * 1.2
            d1 = distance.euclidean(H_a1, coords[N_idx, :])
            d2 = distance.euclidean(H_a2, coords[N_idx, :])
            if d1 < d2:
                vec_a = H_a1
            else:
                vec_a = H_a2
        except ValueError:
            if verb > 0:
                print("Cannot determine the acid vector. return None")
            return None, None, None, None
    else:
        print("Acid substituents are not 3!")
        return None, None, None, None

    vec_a = unit_vector(vec_a)
    H_a = vec_a * 1.2 + coords[B_idx, :]

    return H_a, H_b, subs_acid_idx, subs_base_idx


if __name__ == '__main__':
    xyz1 = '/home/ppdeshmu/test_xyz.xyz'
    test_e = get_dft_E(xyz=xyz1, label='0')
    print(test_e)