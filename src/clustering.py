import os

from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat
import argparse
import numpy as np
from utils.fingerprints import calcfps
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

for file in file_paths:
    name = os.path.basename(file).split('.')[0]
    fps_list = calcfps(file, bitVectObj=True)

    dist_mat = GetTanimotoDistMat(fps_list)
    int_div = np.mean(np.array(dist_mat))
    print(f"Dataset {file} has an internal diversity of {int_div}")

