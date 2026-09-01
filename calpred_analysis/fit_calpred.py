import argparse
import os.path as osp
import sys

import calpred
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "plotting/"))
from plot_utils import ANALYSIS_TO_PHENOTYPE_MAP, BIOBANK_NAME_MAP, MODEL_NAME_MAP
from PRSDataset import PRSDataset


def fit_calpred_model(dataset, standardize=True):
    if standardize:
        dataset.standardize_data()

    results = dict()

    # Extract shared data:
    covars = dataset.get_covariates()
    phenotype = dataset.get_phenotype()

    for prs_id in dataset.prs_ids:
        prs = dataset.data[prs_id].values.reshape(-1, 1)

        sd_data = pd.DataFrame(
            np.concatenate([np.ones_like(prs), prs, covars], axis=1),
            columns=["const", prs_id] + dataset.covariates_cols,
        )

        # For the mean data, also add interaction terms
        # between the PRS and the various covariates:
        mean_data = pd.DataFrame(
            np.concatenate([sd_data.values, prs * covars], axis=1),
            columns=["const", prs_id]
            + dataset.covariates_cols
            + [f"{c}*{prs_id}" for c in dataset.covariates_cols],
        )

        model = calpred.fit(y=phenotype, x=mean_data, z=sd_data)

        results[prs_id] = model.sd_coef

    if standardize:
        dataset.inverse_standardize_data()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit CalPred model to PRS dataset")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the PRS dataset",
    )
    args = parser.parse_args()

    # Read the dataset:
    dataset = PRSDataset.from_pickle(args.dataset)

    analysis_id, biobank = args.dataset.split("/")[-3:-1]

    calpred_output = fit_calpred_model(dataset)

    pgs_names = list(calpred_output.keys())
    output_df = (
        pd.concat([calpred_output[c] for c in pgs_names], axis=1)
        .dropna(axis=0)
        .drop(["const"])
    )
    model_name_map = MODEL_NAME_MAP.get(analysis_id, {})
    output_df.columns = [model_name_map.get(p, p) for p in pgs_names]
    output_df = output_df.loc[["Age", "Sex"] + [f"PC{i + 1}" for i in range(10)], :]

    min_abs_v = np.abs(output_df.values).max()

    sns.set_context("paper", font_scale=1.5)

    sns.heatmap(
        output_df,
        vmin=-min_abs_v,
        vmax=min_abs_v,
        center=0.0,
        cmap="RdBu",
        cbar_kws={"label": "Effect on prediction accuracy ($\\beta_\\sigma$)"},
    )

    pheno_name = ANALYSIS_TO_PHENOTYPE_MAP.get(analysis_id, analysis_id)

    plt.title(f"{pheno_name} ({BIOBANK_NAME_MAP.get(biobank, biobank)})")
    plt.xlabel("Stratified polygenic scores")
    plt.ylabel("Covariates")
    plt.tight_layout()
    plt.savefig(f"figures/calpred/{analysis_id}_{biobank}.eps")
