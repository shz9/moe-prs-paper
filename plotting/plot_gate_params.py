import argparse
import glob
import os.path as osp
import sys

from magenpy.utils.system_utils import makedir

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))

from gate_interpretation import gate_parameters_heatmap
from moe import MoEPRS
from plot_utils import ANALYSIS_TO_PHENOTYPE_MAP, BIOBANK_NAME_MAP_SHORT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot gate parameters for a trained MoE model."
    )
    parser.add_argument(
        "--moe-model",
        dest="moe_model",
        type=str,
        default="MoE-GS",
        help="The name of the MoE model to plot as reference.",
    )

    args = parser.parse_args()

    print(f"> Plotting gate parameters for {args.moe_model}...")

    for f in glob.glob(f"data/trained_models/*/*/train_data/{args.moe_model}.pkl"):
        analysis_id, biobank = f.split("/")[-4:-2]

        phenotype = ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)
        biobank = BIOBANK_NAME_MAP_SHORT.get(biobank, biobank)

        makedir("figures/gate_parameters/")

        title = f"Learned gate parameters for {phenotype} ({biobank})"
        output_f = f"figures/gate_parameters/{analysis_id}_{biobank}.eps"

        print(f"> Processing model: {f}")
        model = MoEPRS.from_saved_model(f)
        gate_parameters_heatmap(
            model, analysis_id, title=title, annot=True, output_file=output_f
        )
