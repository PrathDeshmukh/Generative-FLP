import os
import argparse
from utils.utils import calcfps, calcIntDiv

#Calculate internal diversity first then plot clustering

def parse_cmd():
    parser = argparse.ArgumentParser(description="Calculate internal diversity of datasets")

    parser.add_argument(
        "--file_paths",
        nargs='*'
    )

    return parser.parse_args()

cmd_args = parse_cmd()
file_paths = cmd_args.file_paths
div_dict = {}

for file in file_paths:
    fname = os.path.basename(file).split('.')[0]
    fps_list = calcfps(file, bitVectObj=True)
    div = calcIntDiv(fps_list)
    div_dict[fname] = div
    print(f"Internal diversity of {fname} dataset: {div:.3f}")

with open('Generative-FLP/data/plots/internal_diversity.txt', 'w') as f:
    for fname, div in div_dict.items():
        f.write(f"{fname}: {div:.3f} \n")
