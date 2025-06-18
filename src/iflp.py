import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Get3DDistanceMatrix, rdmolfiles
from scipy.spatial import distance
from skspatial.objects import Plane, Points

from calc_utils import angle_between, get_dft_E, unit_vector, xtb_opt_


def calculate_angle(
    crd_B: int, crd_N: int, cm: np.ndarray, coords: np.ndarray, verb: int = 0
) -> float:
    """
    Calculate the angle between two vectors for a given set of atoms.

    Parameters:
    - crd_B (int): Index of the boron atom.
    - crd_N (int): Index of the nitrogen atom.
    - cm (np.ndarray): Adjacency matrix.
    - coords (np.ndarray): Atomic coordinates.
    - verb (int): Verbosity level.

    Returns:
    float
        The angle in degrees between two vectors, or 0.0 if the calculation cannot be performed.
    """
    subs_base = []
    for n, j in enumerate(cm[crd_N, :]):
        if j == 1:
            pos = coords[n]
            pos = np.around(pos, 4)
            subs_base.append(pos)
    if len(subs_base) == 2:
        if verb > 1:
            print("\nsp2 base. Calculating planes.")
        subs_base.append(coords[crd_N, :])
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            if verb > 0:
                print("Cannot calculate centroid. return 0.0")
            return 0.0
        vec_b = np.array(coords[crd_N, :]) - centroid

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
        vec_b = np.array(coords[crd_N, :]) - centroid
    else:
        print(
            f"Base substituents are not 2 or 3 but rather {len(subs_base)} Angle set to 0!"
        )

    vec_b = unit_vector(vec_b)
    subs_acid = []
    for n, j in enumerate(cm[crd_B, :]):
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
        vec_a = np.array(coords[crd_B, :]) - centroid
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
        crd_B, crd_N = BN_dist_idx[min_dist]
        return min_dist, crd_B, crd_N
    else:
        return None, None, None


def get_H2_pos(
    z: List[int],
    crd_B: int,
    crd_N: int,
    cm: np.ndarray,
    coords: np.ndarray,
    verb: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine cartesian coodinates to which hydride and proton are to be added.

    Vector definition on N: The centroid point of substituent groups - Coordinate of N
    Vector definition on B: Coordinate of H(N) - Coordinate of B

    Set bond distance:
    d(B-H) = 1.2
    d(N-H) = 1.2

    Parameters:
    1. z: vector of atomic numbers
    2. crd_N: atomic index of N
    3. crd_B: atomic index of B
    4. cm: connectivity matrix
    5. coords: coordinates matrix
    6. verb: verbosity level

    Returns:
    H_a, H_b: coordinate of H(B) and H(N), respectively
    """
    # base vector definition
    subs_base = [
        np.around(coords[n], 4)
        for n, j in enumerate(cm[crd_N, :])
        if j == 1 and z[n] != 5
    ]
    if len(subs_base) == 2:
        if verb > 1:
            print("Detect sp2 nitrogen\n")
        subs_base.append(coords[crd_N, :])
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            if verb > 0:
                print("Cannot calculate centroid. return None")
            return None, None
        vec_b = np.array(coords[crd_N, :]) - centroid
    elif len(subs_base) == 3:
        if verb > 1:
            print("Detect sp3 nitrogen\n")
        points = Points(subs_base)
        try:
            centroid = points.centroid()
        except ValueError:
            if verb > 0:
                print("Cannot calculate centroid. return None")
            return None, None

        if distance.euclidean(centroid, coords[crd_N, :]) < 0.1:
            try:
                plane = Plane.best_fit(points)
                n1 = plane.normal
                n2 = -n1
                H_b1 = n1 * 1.2
                H_b2 = n2 * 1.2
                d1 = distance.euclidean(H_b1, coords[crd_B, :])
                d2 = distance.euclidean(H_b2, coords[crd_B, :])
                if d1 < d2:
                    vec_b = H_b1
                else:
                    vec_b = H_b2
            except ValueError:
                if verb > 0:
                    print("Cannot determine the base vector. return None")
                return None, None
        else:
            vec_b = np.array(coords[crd_N, :]) - centroid
    else:
        print(
            f"Base substituents are not 2 or 3 but rather {len(subs_base)} return None"
        )
        return None, None

    vec_b = unit_vector(vec_b)
    H_b = vec_b * 1.2 + coords[crd_N, :]

    # Acid vector definition
    subs_acid = []
    for n, j in enumerate(cm[crd_B, :]):
        if j == 1:
            # if the adjacent atom is not H(which will be the substrate
            # atoms)
            if z[n] != 1:
                pos = coords[n]
                pos = np.around(pos, 4)
                subs_acid.append(pos)

    if len(subs_acid) == 3:
        if verb > 1:
            print("Detect sp3 B center\n")
        points = Points(subs_acid)
        try:
            plane = Plane.best_fit(points)
            n1 = plane.normal
            n2 = -n1
            H_a1 = n1 * 1.2
            H_a2 = n2 * 1.2
            d1 = distance.euclidean(H_a1, coords[crd_N, :])
            d2 = distance.euclidean(H_a2, coords[crd_N, :])
            if d1 < d2:
                vec_a = H_a1
            else:
                vec_a = H_a2
        except ValueError:
            if verb > 0:
                print("Cannot determine the acid vector. return None")
            return None, None
    else:
        print("Acid substituents are not 3!")
        return None, None
    vec_a = unit_vector(vec_a)
    H_a = vec_a * 1.2 + coords[crd_B, :]
    return H_a, H_b


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

        _, crd_B, crd_N = get_shortest_BN_distance(mol_FLP, coords, self.threshold)
        z = [atom.GetAtomicNum() for atom in mol_FLP.GetAtoms()]

        H_a, H_b = get_H2_pos(z, crd_B, crd_N, cm, coords, self.verb)
        if (H_a is None) or (H_b is None):
            if self.verb > 0:
                logging.error(
                    f"Fail to generate H positions for {Chem.MolToSmiles(mol_FLP)}"
                )
            raise ValueError("Fail to generate H positions")
        else:
            init_xyz_path = os.path.join(self.save_folder, f"initial_FLP/{self.label}.xyz")
            rdmolfiles.MolToXYZFile(mol_FLP, init_xyz_path)

            opt_flp_path = xtb_opt_(init_xyz_path, self.save_folder, charge=0)

            #E_iflp = get_dft_E(opt_flp_path, self.label)

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

            mol_BH = rdmolfiles.MolFromXYZFile(opt_BH_path)

            dist_matrix_FLPH2 = Get3DDistanceMatrix(mol_BH)
            ntot = mol_BH.GetNumAtoms()

            d_bh = dist_matrix_FLPH2[ntot - 1, crd_B]
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

            mol_NH = rdmolfiles.MolFromXYZFile(opt_NH_path)

            dist_matrix_FLPH2 = Get3DDistanceMatrix(mol_NH)
            ntot = mol_NH.GetNumAtoms()
            d_nh = dist_matrix_FLPH2[ntot - 1, crd_N]
            if d_nh > 1.5:
                if self.verb > 0:
                    logging.error(
                        f"Fail to generate relaxed FLP-NH for {Chem.MolToSmiles(mol_FLP)}, return 1e10, 1e10"
                    )
                raise ValueError("Fail to generate relaxed FLP-NH")

            #E_iflp_b = get_dft_E(opt_BH_path, self.label, charge=-1)
            #feha = (E_iflp_b - E_iflp - E_HYDRIDE) * h2kcal

            #E_iflp_n = get_dft_E(opt_NH_path, self.label, charge=1)
            #fepa = (E_iflp_n - E_iflp - E_PROTON) * h2kcal
            fepa, feha = 0, 0
        return fepa, feha

    def geom_targets(self) -> List[float]:
        """
        Calculate geometric properties of a relaxed molecule.

        Parameters:
        - mol_relaxed (Chem.Mol): The relaxed molecule for which geometric properties will be calculated.

        Returns:
        Tuple[float, float]: A tuple containing the best B-N distance and the angle (eta) in degrees.
        If the calculation fails, an appropriate exception is raised.
        """
        try:
            mol_relaxed = Chem.MolFromXYZFile(self.mol_relaxed_xyz)
            cm = Chem.rdmolops.GetAdjacencyMatrix(mol_relaxed)

            conf = mol_relaxed.GetConformer()
            coords = conf.GetPositions()

            best_dist, crd_B, crd_N = get_shortest_BN_distance(
                mol_relaxed, coords, self.threshold
            )
            eta = calculate_angle(crd_B, crd_N, cm, coords, self.verb)
            return best_dist, eta
        except Exception as e:
            raise ValueError(f"Fail to compute geometric properties: {str(e)}") from e
