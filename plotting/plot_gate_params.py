import sys
import os.path as osp
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))
from plot_utils import BIOBANK_NAME_MAP_SHORT, PHENOTYPE_NAME_MAP
import argparse
from magenpy.utils.system_utils import makedir
from gate_interpretation import gate_parameters_heatmap
from moe import MoEPRS
import glob


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot gate parameters for a trained MoE model.')
    parser.add_argument('--moe-model', dest='moe_model',
                        type=str, default='MoE-global-int',
                        help="The name of the MoE model to plot as reference.")

    args = parser.parse_args()

    print(f"> Plotting gate parameters for {args.moe_model}...")

    for f in glob.glob(f"data/trained_models/*/*/train_data/{args.moe_model}.pkl"):

        phenotype_code, biobank = f.split('/')[-4:-2]

        phenotype = PHENOTYPE_NAME_MAP.get(phenotype_code, phenotype_code)
        biobank = BIOBANK_NAME_MAP_SHORT.get(biobank, biobank)

        makedir(f"figures/gate_parameters/")

        title = f"Learned gate parameters for {phenotype} ({biobank})"
        output_f = f"figures/gate_parameters/{phenotype_code}_{biobank}.eps"

        print(f"> Processing model: {f}")
        model = MoEPRS.from_saved_model(f)
        gate_parameters_heatmap(model,
                                title=title,
                                annot=True,
                                output_file=output_f)

