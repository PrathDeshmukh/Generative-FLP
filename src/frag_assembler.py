import pandas as pd
import random
import argparse

def chromosome_to_smiles():
    def sc2smiles(chromosome):
        """Generate a smiles string from a list of SMILES fragments."""
        LAr1 = chromosome[0]
        LAr2 = chromosome[1]
        LBr1 = chromosome[2]
        LBr2 = chromosome[3]
        BB1 = chromosome[4]
        BBrA = chromosome[5]
        BBrB = chromosome[6]
        BBrC = chromosome[7]

        def external_FLP(smiles_list, BB1, LBr1, LBr2):
            smiles_list.append(BB1)
            LBr = [LBr1, LBr2]
            if any("*" in LB for LB in LBr):
                lb = filter(lambda LB: "*" in LB in LBr, LBr)
                lb = list(lb)[0]
                if "c1" in lb:
                    smiles_list.append("n1")
                    smiles_list.append(f"({lb[1:]})")
                else:
                    smiles_list.append("N1")
                    smiles_list.append(f"({lb[1:]})")
            else:
                smiles_list.append("N")
                smiles_list.extend([f"({LBr1})", f"({LBr2})"])
            return smiles_list

        smiles_list = []
        LAr = [LAr1, LAr2]
        # addressing Catalyst-A-LA
        if any("*" in LA for LA in LAr):
            la = filter(lambda LA: "*" in LA in LAr, LAr)
            la = list(la)[0]
            smiles_list.append("B1")
            smiles_list.append(f"({la[1:]})")
        else:
            smiles_list.append("B")
            smiles_list.extend([f"({LAr1})", f"({LAr2})"])

        # adding substituent groups to BB
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

            # LBr2 must be null here, as well as LBr1 if it is *.
            # 2B and 2A
            if "*" in BB1_str:
                smiles_list.append(BB1_str[1:])
                smiles_list.append("N1")
                if not ("*" in LBr1) and (BB1_str[-1] != "="):
                    smiles_list.append(f"({LBr1})")
            # 2D
            elif "@" in BB1_str and "$" in BB1_str:
                smiles_list.append(BB1_str[2:])
                smiles_list.append("n1")
            # 2C
            elif "@" in BB1_str:
                smiles_list.append(BB1_str[1:])
                smiles_list.append("N12")

            # External FLP
            else:
                smiles_list = external_FLP(smiles_list, BB1_str, LBr1, LBr2)

        else:
            smiles_list = external_FLP(smiles_list, BB1, LBr1, LBr2)

        attempt_smiles = "".join(smiles_list)
        # print("Input SMILES: ", attempt_smiles)

        return attempt_smiles
        # Recommended sanitization
        # mol, smiles, ok = sanitize_smiles(attempt_smiles)

    # if ok and mol is not None:
    #     return smiles
    # else:
    #     print(f"Generated SMILES string {attempt_smiles} could not be sanitized.")
    #     raise ValueError

    return sc2smiles


if __name__== "__main__":

    def parse_cmd():
        parser = argparse.ArgumentParser(description="Generation of random SMILES string")

        parser.add_argument(
            "--frag_path",
            type=str,
            default="/home/pratham/PycharmProjects/ga-flp-ruben/database_HC_bbr.csv"
        )

        parser.add_argument(
            "--n_smiles",
            type=int,
            default=500
        )

        return parser.parse_args()

    cmd_args = parse_cmd()
    frag_csv = cmd_args.frag_path
    n_smiles = cmd_args.n_smiles

    random.seed(42)

    frag_df = pd.read_csv(frag_csv)
    column_names = list(frag_df)
    smiles_dict = {cname: list(frag_df[cname].dropna()) for cname in column_names}

    assembled_smiles = []

    i = 0
    while i < n_smiles:

        chromosome = []
        for frag_type, smiles_list in smiles_dict.items():

            if frag_type == 'BB':
                chromosome.append(random.choices(smiles_list, k=1))

            elif frag_type == 'LAr' or frag_type == 'LBr':
                chromosome.append(random.choices(smiles_list, k=2))

            elif frag_type == 'BBr':
                chromosome.append(random.choices(smiles_list, k=3))

        # random.choice returns a list so we need to unpack it
        chromosomes = []
        chromosomes.extend([c for c_list in chromosome for c in c_list])

        smiles = chromosome_to_smiles()(chromosomes)
        assembled_smiles.append(smiles)

        i += 1

    assembled_smiles_df = pd.DataFrame(assembled_smiles, columns=['SMILES'])
    assembled_smiles_df.to_csv(f'./assembled-smiles-{n_smiles}.csv')