import logging
import os
import shutil
import subprocess as sp
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Get3DDistanceMatrix, rdmolfiles
from scipy.spatial import distance
from skspatial.objects import Plane, Points

from src.calc_utils import angle_between, cleanup_mol, get_dft_E, unit_vector, xtb_opt, xtb_opt_
from src.smiles_cleaning import save_path


def chromosome_to_smiles():
    """
    Convert a chromosome into an IFLP SMILES string.

    Parameters:
    -----------
    chromosome : list
        A list containing the components of the chromosome in the order: [LAr1, LAr2, LBr1, LBr2, BB1, BBrA, BBrB, BBrC]

    Returns:
    --------
    str
        The SMILES string generated from the chromosome.
    """

    def exo_exo_IFLP(smiles_list, BB1, LBr1, LBr2):
        smiles_list.append(BB1)
        LBr = [LBr1, LBr2]
        if any("*" in LB for LB in LBr):
            lb = filter(lambda LB: "*" in LB in LBr, LBr)
            lb = list(lb)[0]
            if "c1" in lb:
                smiles_list.extend(["n1", f"({lb[1:]})"])
            else:
                smiles_list.extend(["N1", f"({lb[1:]})"])
        else:
            smiles_list.extend(["N", f"({LBr1})", f"({LBr2})"])
        return smiles_list

    def sc2smiles(chromosome):
        LAr1, LAr2, LBr1, LBr2, BB1, BBrA, BBrB, BBrC = chromosome
        smiles_list = []
        LAr = [LAr1, LAr2]

        if any("*" in LA for LA in LAr):
            la = filter(lambda LA: "*" in LA in LAr, LAr)
            la = list(la)[0]
            smiles_list.extend(["B1", f"({la[1:]})"])
        else:
            smiles_list.extend(["B", f"({LAr1})", f"({LAr2})"])

        if "()" in BB1:
            pieces = BB1.split("()")
            BB1 = []
            for i, piece in enumerate(pieces[:-1]):
                BB1.append(piece)
                if i == 0:
                    BB1.append(f"({BBrA})")
                elif i == len(pieces[:-1]) - 1:
                    BB1.append(f"({BBrB})")
                else:
                    BB1.append(f"({BBrC})")
            BB1.append(pieces[-1])
            BB1_str = "".join(BB1)

            if "*" in BB1_str:
                smiles_list.extend([BB1_str[1:], "N1"])
                if not ("*" in LBr1) and (BB1_str[-1] != "="):
                    smiles_list.append(f"({LBr1})")
            elif "@" in BB1_str and "$" in BB1_str:
                smiles_list.extend([BB1_str[2:], "n1"])
            elif "@" in BB1_str:
                smiles_list.extend([BB1_str[1:], "N12"])
            else:
                smiles_list = exo_exo_IFLP(smiles_list, BB1_str, LBr1, LBr2)

        else:
            smiles_list = exo_exo_IFLP(smiles_list, BB1, LBr1, LBr2)

        attempt_smiles = "".join(smiles_list)
        return attempt_smiles

    return sc2smiles


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
    mol: Chem.Mol, dist_matrix: np.ndarray, threshold: float
) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Calculate the shortest B-N distance above a specified threshold.

    Parameters:
    - mol (Chem.Mol): The molecular structure.
    - dist_matrix (np.ndarray): The distance matrix.
    - threshold (float): The threshold for considering B-N distances.

    Returns:
    tuple: A tuple containing the following:
        - float or None: The shortest B-N distance greater than the threshold, or None if no such distance is found.
        - int or None: The index of the boron (B) atom corresponding to the shortest distance, or None if not found.
        - int or None: The index of the nitrogen (N) atom corresponding to the shortest distance, or None if not found.
    """
    N_index = []
    B_index = []

    for i, atom in enumerate(mol.GetAtoms(), start=1):
        if atom.GetAtomicNum() == 5:
            B_index.append(i)
        elif atom.GetAtomicNum() == 7:
            N_index.append(i)

    BN_list = np.asarray([dist_matrix[i - 1, j - 1] for j in B_index for i in N_index])

    valid_distances = BN_list[BN_list > threshold]

    if valid_distances.size > 0:
        best_dist = np.min(valid_distances)
        bn_idxs = np.where(dist_matrix == best_dist)
        crd_N = bn_idxs[0][0] if bn_idxs[0].size > 0 else None
        crd_B = bn_idxs[1][0] if bn_idxs[1].size > 0 else None
        return best_dist, crd_B, crd_N
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
            # if the adjacant atom is not H(which will be the substrate
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
            d1 = distance.euclidean(H_a1, coords[crd_B, :])
            d2 = distance.euclidean(H_a2, coords[crd_B, :])
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

    def get_IFLP(self, smiles: str) -> Chem.Mol:
        """
        Convert a SMILES string representing an IFLP compound into an RDKit molecule object,
        optimized at the desired GFN Hamiltonian level.

        Parameters:
        -----------
        smiles : str
            The SMILES string representing the internal FLP compound.

        Returns:
        --------
        Chem.Mol
            An RDKit molecule object for the optimized FLP compound.
        """

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        mol = cleanup_mol(mol, n_confs=10)

        max_iterations = 10
        for i in range(2, max_iterations + 1):
            if mol.GetNumConformers() != 0:
                break
            mol = cleanup_mol(mol, n_confs=10 + i * 5)
        else:
            raise ValueError(f"Cannot generate mol object for {smiles}")

        file_name = f"Hoshiyomi_{self.label}.xyz"

        rdmolfiles.MolToXYZFile(mol, file_name)

        if os.path.getsize(file_name) == 0:
            logging.error(f"Cannot generate mol object for {smiles}, return None")
            return None
        xtb_opt(file_name, level=self.level)
        mol_FLP_s = Chem.rdmolfiles.MolFromMolFile(
            "xtbtopo.mol", removeHs=False, sanitize=False
        )
        os.remove("xtbtopo.mol")
        return mol_FLP_s

    def gen_relaxed_FLP_H2(self) -> Tuple[Optional[Chem.Mol], Optional[Chem.Mol]]:
        """
        Generate a relaxed IFLP (HB---NH intermediate without hydrogen).

        Parameters:
        - smiles (str): The SMILES representation of IFLP molecule.

        Returns:
        Tuple[Chem.Mol or None, Chem.Mol or None]: A tuple containing the relaxed FLP-H2 molecule
        as an RDKit Mol object and the original IFLP molecule, both as optional values.
        If generation fails, (None, None) is returned for both.
        """

        mol_FLP = self.init_mol_obj
        smiles = Chem.MolToSmiles(mol_FLP)
        try:
            if mol_FLP is None:
                return None
            dist_matrix = Get3DDistanceMatrix(mol_FLP)
        except ValueError as e:
            if self.verb > 0:
                logging.error(
                    f"Fail to generate mol and/or distance matrix for {smiles}, return None"
                )
            return None, None

        _, crd_B, crd_N = get_shortest_BN_distance(mol_FLP, dist_matrix, self.threshold)
        cm = Chem.rdmolops.GetAdjacencyMatrix(mol_FLP)
        z = [atom.GetAtomicNum() for atom in mol_FLP.GetAtoms()]

        conf = mol_FLP.GetConformer()
        coords = conf.GetPositions()

        H_a, H_b = get_H2_pos(z, crd_B, crd_N, cm, coords, self.verb)
        if (H_a is None) or (H_b is None):
            if self.verb > 0:
                logging.error(f"Fail to generate H positions for {smiles}, return None")
            return None, None
        else:
            rdmolfiles.MolToXYZFile(mol_FLP, f"Hoshiyomi_{self.label}.xyz")
            ha_pos = f"H  {H_a[0]}   {H_a[1]}   {H_a[2]}\n"
            hb_pos = f"H  {H_b[0]}   {H_b[1]}   {H_b[2]}"

            with open(f"Hoshiyomi_{self.label}.xyz", "r") as f:
                contents = f.readlines()
            contents.extend([ha_pos, hb_pos])
            contents[0] = str(len(contents[2:])) + "\n"

            with open(f"Hoshiyomi_FLP_H2_{self.label}.xyz", "w") as f:
                f.writelines(contents)

            xtb_opt(f"Hoshiyomi_FLP_H2_{self.label}.xyz", level=self.level)

            mol_FLP_H2 = Chem.rdmolfiles.MolFromMolFile(
                "xtbtopo.mol", removeHs=False, sanitize=False
            )
            dist_matrix_FLPH2 = Get3DDistanceMatrix(mol_FLP_H2)
            ntot = mol_FLP_H2.GetNumAtoms()
            d_bh = dist_matrix_FLPH2[ntot - 2, crd_B]
            d_nh = dist_matrix_FLPH2[ntot - 1, crd_N]

            if (d_bh > 1.5) or (d_nh > 1.5):
                if self.verb > 0:
                    logging.error(
                        f"Fail to generate relaxed FLP-H2 for {smiles}, return None"
                    )
                return None, None

            with open(f"Hoshiyomi_FLP_H2_{self.label}_xtbopt.xyz", "r") as f:
                contents = f.readlines()

            with open(f"Hoshiyomi_relax_{self.label}.xyz", "w") as f:
                f.write(str(len(z)) + "\n\n")
                f.writelines(contents[2:-2])

            sp.call(
                ["xtb", f"Hoshiyomi_relax_{self.label}.xyz"],
                stdout=sp.DEVNULL,
                stderr=sp.STDOUT,
            )

            mol_FLP_relaxed = Chem.rdmolfiles.MolFromMolFile(
                "xtbtopo.mol", removeHs=False, sanitize=False
            )
        return mol_FLP_relaxed, mol_FLP

    def gen_relaxed_FLP_H2_xyz(
        self,
        mol_FLP: Chem.Mol,
        name_out: str = "Hoshiyomi_relax.xyz",
        optimization: bool = False,
    ) -> int:
        try:
            dist_matrix = Get3DDistanceMatrix(mol_FLP)
        except ValueError as e:
            if self.verb > 0:
                print(f"Fail to generate mol and/or distance matrix, return 0")
            return 0
        _, crd_B, crd_N = get_shortest_BN_distance(mol_FLP, dist_matrix, self.threshold)

        cm = Chem.rdmolops.GetAdjacencyMatrix(mol_FLP)
        z = [atom.GetAtomicNum() for atom in mol_FLP.GetAtoms()]
        for c in mol_FLP.GetConformers():
            coords = c.GetPositions()

        H_a, H_b = get_H2_pos(z, crd_B, crd_N, cm, coords, self.verb)
        if (H_a is None) or (H_b is None):
            if self.verb > 0:
                print(f"Fail to generate H positions, return 0")
            return 0
        else:
            rdmolfiles.MolToXYZFile(mol_FLP, f"Hoshiyomi_{self.label}.xyz")
            ha_pos = f"H  {H_a[0]}   {H_a[1]}   {H_a[2]}\n"
            hb_pos = f"H  {H_b[0]}   {H_b[1]}   {H_b[2]}"

            with open(f"Hoshiyomi_{self.label}.xyz", "r") as f:
                contents = f.readlines()
            contents.extend([ha_pos, hb_pos])
            contents[0] = str(len(contents[2:])) + "\n"

            with open(name_out, "w") as f:
                f.writelines(contents)
            if os.path.exists(f"Hoshiyomi_{self.label}.xyz"):
                os.remove(f"Hoshiyomi_{self.label}.xyz")
            if optimization:
                xtb_opt(name_out, level=self.level)
                shutil.move(f"{name_out[:-4]}_xtbopt.xyz", name_out)
                return 1
            return 1

    def gen_FLP_NH_xyz(
        self,
        mol_FLP: Chem.Mol,
        name_out: str = "Hoshiyomi_NH.xyz",
        optimization: bool = False,
    ) -> int:
        """
        Generate XYZ file for FLP-NH molecule.

        Parameters:
        - mol_FLP (Chem.Mol): The FLP molecule.
        - name_out (str): Output file name. Default is "Hoshiyomi_NH.xyz".
        - optimization (bool): Perform optimization or not.

        Returns:
        int: 1 if successful, 0 otherwise.
        """
        try:
            dist_matrix = Get3DDistanceMatrix(mol_FLP)
        except ValueError as e:
            if self.verb > 0:
                logging.error("Fail to generate mol and/or distance matrix, return 0")
            return 0
        _, crd_B, crd_N = get_shortest_BN_distance(mol_FLP, dist_matrix, self.threshold)

        cm = Chem.rdmolops.GetAdjacencyMatrix(mol_FLP)
        z = [atom.GetAtomicNum() for atom in mol_FLP.GetAtoms()]
        for c in mol_FLP.GetConformers():
            coords = c.GetPositions()

        H_a, H_b = get_H2_pos(z, crd_B, crd_N, cm, coords, self.verb)
        if (H_a is None) or (H_b is None):
            if self.verb > 0:
                logging.error("Fail to generate H positions, return 0")
            return 0
        else:
            xyz_file = f"Hoshiyomi_{self.label}.xyz"
            rdmolfiles.MolToXYZFile(mol_FLP, xyz_file)
            hb_pos = f"H  {H_b[0]}   {H_b[1]}   {H_b[2]}\n"

            with open(xyz_file, "r") as f:
                contents = f.readlines()
            contents.append(hb_pos)
            contents[0] = str(len(contents[2:])) + "\n"

            with open(name_out, "w") as f:
                f.writelines(contents)
            if os.path.exists(xyz_file):
                os.remove(xyz_file)

            if optimization:
                xtb_opt(name_out, level=self.level, charge=1)
                shutil.move(f"{name_out[:-4]}_xtbopt.xyz", name_out)
                return 1
            return 1

    def gen_FLP_BH_xyz(
        self,
        mol_FLP: Chem.Mol,
        name_out: str = "Hoshiyomi_BH.xyz",
        optimization: bool = False,
    ) -> int:
        """
        Generate XYZ file for FLP-BH molecule.

        Parameters:
        - mol_FLP (Chem.Mol): The FLP molecule.
        - name_out (str): Output file name. Default is "Hoshiyomi_BH.xyz".
        - optimization (bool): Perform optimization or not.

        Returns:
        int: 1 if successful, 0 otherwise.
        """
        try:
            dist_matrix = Get3DDistanceMatrix(mol_FLP)
        except ValueError:
            if self.verb > 0:
                logging.error(f"Fail to generate mol and/or distance matrix, return 0")
            return 0
        _, crd_B, crd_N = get_shortest_BN_distance(mol_FLP, dist_matrix, self.threshold)

        cm = Chem.rdmolops.GetAdjacencyMatrix(mol_FLP)
        z = [atom.GetAtomicNum() for atom in mol_FLP.GetAtoms()]
        for c in mol_FLP.GetConformers():
            coords = c.GetPositions()

        H_a, H_b = get_H2_pos(z, crd_B, crd_N, cm, coords, self.verb)
        if (H_a is None) or (H_b is None):
            if self.verb > 0:
                logging.error(f"Fail to generate H positions, return 0")
            return 0
        else:
            rdmolfiles.MolToXYZFile(mol_FLP, f"Hoshiyomi_{self.label}.xyz")
            ha_pos = f"H  {H_a[0]}   {H_a[1]}   {H_a[2]}\n"

            with open(f"Hoshiyomi_{self.label}.xyz", "r") as f:
                contents = f.readlines()
            contents.append(ha_pos)
            contents[0] = str(len(contents[2:])) + "\n"

            with open(name_out, "w") as f:
                f.writelines(contents)
            if os.path.exists(f"Hoshiyomi_{self.label}.xyz"):
                os.remove(f"Hoshiyomi_{self.label}.xyz")
            if optimization:
                xtb_opt(name_out, level=self.level, charge=-1)
                shutil.move(f"{name_out[:-4]}_xtbopt.xyz", name_out)
                return 1
            return 1

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

        E_PROTON = 0.229552166667
        E_HYDRIDE = -0.610746694539
        h2kcal = 627.509

        mol_FLP = Chem.AddHs(mol_FLP)
        AllChem.EmbedMolecule(mol_FLP)

        conf = mol_FLP.GetConformer()
        coords = conf.GetPositions()

        dist_matrix = Get3DDistanceMatrix(mol_FLP)
        cm = Chem.rdmolops.GetAdjacencyMatrix(mol_FLP)

        _, crd_B, crd_N = get_shortest_BN_distance(mol_FLP, dist_matrix, self.threshold)
        z = [atom.GetAtomicNum() for atom in mol_FLP.GetAtoms()]

        H_a, H_b = get_H2_pos(z, crd_B, crd_N, cm, coords, self.verb)
        if (H_a is None) or (H_b is None):
            if self.verb > 0:
                logging.error(
                    f"Fail to generate H positions for {Chem.MolToSmiles(mol_FLP)}"
                )
            raise ValueError("Fail to generate H positions")
        else:
            init_xyz_path = os.path.join(self.save_folder, f"/initial_FLP/{self.label}.xyz")
            rdmolfiles.MolToXYZFile(mol_FLP, init_xyz_path)

            opt_flp_path = xtb_opt_(init_xyz_path, self.save_folder, charge=0)

            E_iflp = get_dft_E(opt_flp_path, self.label)

            ha_pos = f"H  {H_a[0]}   {H_a[1]}   {H_a[2]}\n"
            hb_pos = f"H  {H_b[0]}   {H_b[1]}   {H_b[2]}"

            with open(init_xyz_path, "r") as f:
                contents = f.readlines()

            contents_b = contents.copy()
            contents_c = contents.copy()

            #Generate and relax intermediate 2 of FLP
            contents_c.append(ha_pos)
            contents_c.append(hb_pos)

            flp_h2_path = os.path.join(self.save_folder, f"/initial_FLP/FLP_H2_{self.label}.xyz")
            with open(flp_h2_path, "w") as f:
                f.writelines(contents_c)

            opt_flp_h2_path = xtb_opt_(flp_h2_path, self.save_folder, charge=0)
            self.mol_relaxed_xyz = opt_flp_h2_path

            # FEHA
            contents.append(ha_pos)
            contents[0] = str(len(contents[2:])) + "\n"

            flp_BH_path = os.path.join(self.save_folder, f"/initial_FLP/FLP_BH_{self.label}.xyz")

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

            E_iflp_b = get_dft_E(opt_BH_path, self.label, charge=-1)
            feha = (E_iflp_b - E_iflp - E_HYDRIDE) * h2kcal

            # FEPA
            contents_b.append(hb_pos)
            contents_b[0] = str(len(contents_b[2:])) + "\n"

            flp_NH_path = os.path.join(self.save_folder, f"/initial_FLP/FLP_NH_{self.label}.xyz")

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

            E_iflp_n = get_dft_E(opt_NH_path, self.label, charge=1)
            fepa = (E_iflp_n - E_iflp - E_PROTON) * h2kcal
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
            dist_matrix = Get3DDistanceMatrix(mol_relaxed)
            cm = Chem.rdmolops.GetAdjacencyMatrix(mol_relaxed)

            conf = mol_relaxed.GetConformer()
            coords = conf.GetPositions()

            best_dist, crd_B, crd_N = get_shortest_BN_distance(
                mol_relaxed, dist_matrix, self.threshold
            )
            eta = calculate_angle(crd_B, crd_N, cm, coords, self.verb)
            return best_dist, eta
        except Exception as e:
            raise ValueError(f"Fail to compute geometric properties: {str(e)}") from e
