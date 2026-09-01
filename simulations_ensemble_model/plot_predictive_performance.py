import argparse
import glob
import json
import os
import os.path as osp
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from magenpy.utils.system_utils import makedir

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "score/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))

from baseline_models import AncestryWeightedPRS, MultiPRS
from evaluate_predictive_performance import stratified_evaluation
from moe import MoEPRS
from PRSDataset import PRSDataset


def map_sim_scenario_names(col):
    return col.map(
        {
            "single_model": "Single model",
            "multiprs": "MultiPRS",
            "discrete_context (Sex)": "Discrete context (Sex)",
            "discrete_context (Ancestry)": "Discrete context (Ancestry)",
            "continuous_context (Age)": "Continuous context (Age)",
            "moe": "Mixture-of-Experts",
        }
    ).fillna(col)


def get_sim_order(sims):
    return [
        s
        for s in [
            "Single model",
            "MultiPRS",
            "Mixture-of-Experts",
            "Discrete context (Sex)",
            "Discrete context (Ancestry)",
            "Continuous context (Age)",
        ]
        if s in sims
    ]


def extract_trained_models(dataset_path, model_subset=None):
    """
    For a given dataset path, extract trained models from the specified model subset.
    """

    trained_models_path = osp.dirname(
        dataset_path.replace("harmonized_data", "trained_models")
    )
    trained_models_path = osp.join(trained_models_path, "*", "*.pkl")

    trained_models = {}

    for f in glob.glob(trained_models_path):
        model_name = osp.basename(f).replace(".pkl", "")

        if model_subset is not None:
            if model_name not in model_subset:
                continue

        if "moe" in model_name.lower():
            trained_models[model_name] = MoEPRS.from_saved_model(f)
        else:
            trained_models[model_name] = MultiPRS.from_saved_model(f)

    if len(trained_models) == 0:
        raise FileNotFoundError(f"No trained models found in {trained_models_path}")

    return trained_models


def _normalize_predictive_eval_result(eval_result, simulation_config):
    """
    Convert either legacy wide evaluation output or the current long-format
    evaluation output into the plotting schema used by this script.
    """

    model_name_map = {
        "MoE": "MoEPRS",
        "MoE-global-int": "MoEPRS",
        "MultiPRS": "MultiPRS",
    }

    if {"model_name", "metric", "value"}.issubset(eval_result.columns):
        eval_result = eval_result.loc[
            (eval_result["model_name"].isin(model_name_map))
            & (eval_result["metric"] == "Incremental_R2")
            & (eval_result["metric_kind"] == "base")
            & (eval_result["eval_category"] == "All")
            & (eval_result["eval_group"] == "All")
        ].copy()

        eval_result["Model"] = eval_result["model_name"].map(model_name_map)
        eval_result["Incremental_R2"] = eval_result["value"]
        eval_result = eval_result[["Model", "Incremental_R2"]]

    elif {"PGS", "Incremental_R2"}.issubset(eval_result.columns):
        eval_result = eval_result.loc[eval_result["PGS"].isin(model_name_map)].copy()
        eval_result["Model"] = eval_result["PGS"].map(model_name_map)
        eval_result = eval_result[["Model", "Incremental_R2"]]

    else:
        raise ValueError(
            "Unsupported evaluation output format. Expected current long-format "
            "columns ('model_name', 'metric', 'value') or legacy columns "
            "('PGS', 'Incremental_R2')."
        )

    eval_result["Heritability"] = simulation_config["heritability"]
    eval_result["Simulation Scenario"] = simulation_config["simulation_type"]

    return eval_result


def evaluate_prediction_accuracy_on_dataset(dataset_path):
    print("Evaluating:", dataset_path)
    prs_dataset = PRSDataset.from_pickle(dataset_path)

    # Load the simulation configuration from the pickle file:
    config_path = osp.join(osp.dirname(dataset_path), "config.pkl")
    with open(config_path, "rb") as f:
        simulation_config = pickle.load(f)

    # Load the models that were trained on this dataset:
    trained_models = extract_trained_models(
        dataset_path,
        model_subset=["MoE", "MultiPRS"],
    )

    eval_result = stratified_evaluation(prs_dataset, trained_models)

    return _normalize_predictive_eval_result(eval_result, simulation_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictive performance")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs to launch when performing evaluation",
    )
    parser.add_argument(
        "--analysis-id",
        "--phenotype",
        dest="analysis_id",
        type=str,
        default="HEIGHT_MA",
        help="Analysis ID to plot simulation results for. --phenotype is kept as a legacy alias.",
    )
    parser.add_argument(
        "--biobank",
        type=str,
        default="ukbb",
        help="Biobank ID to plot simulation results for.",
    )
    args = parser.parse_args()

    predictive_perf = []

    dataset_paths = glob.glob(
        f"data/harmonized_data_simulations/sim_*/{args.analysis_id}/{args.biobank}/*_h0.*/test_data.pkl"
    )
    if len(dataset_paths) == 0:
        raise FileNotFoundError(
            "No simulation test datasets found for "
            f"analysis_id={args.analysis_id}, biobank={args.biobank}."
        )

    predictive_perf = Parallel(n_jobs=args.jobs, backend="multiprocessing")(
        delayed(evaluate_prediction_accuracy_on_dataset)(path) for path in dataset_paths
    )

    predictive_perf = pd.concat(predictive_perf)
    predictive_perf["Simulation Scenario"] = map_sim_scenario_names(
        predictive_perf["Simulation Scenario"]
    )
    predictive_perf.rename(
        columns={"Simulation Scenario": "Scenario"}, inplace=True
    )

    sns.set_context("paper", font_scale=2.25)

    g = sns.catplot(
        data=predictive_perf,
        x="Heritability",
        col="Scenario",
        col_order=get_sim_order(predictive_perf["Scenario"].unique()),
        hue_order=["MoEPRS", "MultiPRS"],
        col_wrap=3,
        y="Incremental_R2",
        kind="box",
        showfliers=False,
        hue="Model",
        palette={"MoEPRS": "#375E97", "MultiPRS": "#FFBB00"},
    )

    for ax in g.axes.flat:
        title = ax.get_title()
        if title.startswith("Scenario = "):
            ax.set_title(title.replace("Scenario = ", ""))

    g.set_ylabels("Incremental $R^2$")

    plt.savefig(
        f"figures/simulations/predictive_performance_{args.analysis_id}_{args.biobank}.eps"
    )
