import argparse
import glob
import os.path as osp
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, roc_auc_score

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "score/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))
from moe import MoEPRS
from plot_predictive_performance import (
    extract_trained_models,
    get_sim_order,
    map_sim_scenario_names,
)
from PRSDataset import PRSDataset


def cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-15,
    normalize: bool = True,
) -> float:
    """
    Compute cross entropy loss for soft or hard labels.

    Compatible with sklearn's metric interface and handles both discrete
    labels and probability distributions as ground truth.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        Ground truth labels. Can be:
        - 1D array of class indices (hard labels)
        - 2D array of probability distributions (soft labels)

    y_pred : array-like of shape (n_samples, n_classes)
        Predicted probabilities for each class.

    epsilon : float, default=1e-15
        Small value to clip predictions and avoid log(0).

    normalize : bool, default=True
        If True, return mean cross entropy per sample.
        If False, return sum of cross entropy.

    Returns
    -------
    loss : float
        Cross entropy loss.

    Examples
    --------
    >>> # Hard labels (discrete)
    >>> y_true = np.array([0, 1, 2])
    >>> y_pred = np.array([[0.7, 0.2, 0.1],
    ...                     [0.1, 0.8, 0.1],
    ...                     [0.2, 0.2, 0.6]])
    >>> cross_entropy(y_true, y_pred)
    0.4337...

    >>> # Soft labels (probability distributions)
    >>> y_true = np.array([[0.8, 0.1, 0.1],
    ...                     [0.0, 0.9, 0.1],
    ...                     [0.1, 0.1, 0.8]])
    >>> cross_entropy(y_true, y_pred)
    0.4969...
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Handle hard labels (1D array of class indices)
    if y_true.ndim == 1:
        n_samples = y_true.shape[0]
        # Convert to one-hot encoding
        n_classes = y_pred.shape[1]
        y_true_one_hot = np.zeros((n_samples, n_classes))
        y_true_one_hot[np.arange(n_samples), y_true.astype(int)] = 1
        y_true = y_true_one_hot

    # Compute cross entropy: -sum(y_true * log(y_pred))
    ce = -np.sum(y_true * np.log(y_pred), axis=1)

    if normalize:
        return np.mean(ce)
    else:
        return np.sum(ce)


def extract_true_proba(dataset, config):
    proba = np.zeros(shape=(dataset.N, len(dataset.prs_ids)))

    if config["simulation_type"] == "single_model":
        proba[:, dataset.prs_ids.index(config["single_model_id"])] = 1.0

        np.repeat(
            dataset.prs_ids.index(config["single_model_id"]),
            dataset.N,
        )

    elif "discrete_context" in config["simulation_type"]:
        context_idx = dataset.data[config["context"]].map(config["context_to_scores"])
        proba[np.arange(proba.shape[0]), context_idx] = 1.0

    elif config["simulation_type"] == "multiprs":
        proba = np.repeat(
            config["mixing_weights"][np.newaxis, :], proba.shape[0], axis=0
        )
    elif config["simulation_type"] == "moe":
        covars = dataset.get_covariates()
        # Scale covariates similar to data generative process:
        covars = config["covar_scaler"].transform(covars)
        # add intercept:
        covars = np.hstack([np.ones((covars.shape[0], 1)), covars])

        # Generate random gating weights:
        gate_weights = config["gate_weights"]

        # Generate the mixing weights for each individual:
        from scipy.special import softmax

        proba = softmax(covars.dot(gate_weights), axis=1)
    elif "continuous_context" in config["simulation_type"]:
        # Get the context variable and standardize it:
        ctx = dataset.data[config["context"]].values
        ctx = (ctx - config["context_statistics"]["mean"]) / config[
            "context_statistics"
        ]["std"]

        w0, w1 = config["sigmoid_weights"]
        from scipy.special import expit

        w = expit(w0 + w1 * ctx)

        pgs_idx1, pgs_idx2 = config["selected_scores"]
        proba[:, pgs_idx1] = w
        proba[:, pgs_idx2] = 1.0 - w

    else:
        raise ValueError(
            "Simulation type evaluation not supported: " + config["simulation_type"]
        )

    return proba


def evaluate_preds(true_proba, pred_proba):
    """
    Evaluate the predictions of gating models.
    """

    scores = {
        "Cross entropy": cross_entropy(true_proba, pred_proba),
        "Brier score": np.mean(np.sum((true_proba - pred_proba) ** 2, axis=1)),
        "Accuracy": accuracy_score(
            true_proba.argmax(axis=1), pred_proba.argmax(axis=1)
        ),
    }

    return scores


def evaluate_dataset(dataset_path):
    print("Evaluating:", dataset_path)

    dataset = PRSDataset.from_pickle(dataset_path)

    # Read simulation configuration file:
    config_file = dataset_path.replace("test_data.pkl", "config.pkl")
    with open(config_file, "rb") as f:
        config = pickle.load(f)

    moe_models = extract_trained_models(
        dataset_path, model_subset=["MoE", "MoE-CFG"]
    )

    true_proba = extract_true_proba(dataset, config)

    metrics = []

    for model_name, moe_model in moe_models.items():
        # Get the predicted probabilities:
        pred_proba = moe_model.predict_proba(dataset)

        metrics.append(evaluate_preds(true_proba, pred_proba))

        metrics[-1].update(
            {
                "Heritability": config["heritability"],
                "Simulation Scenario": config["simulation_type"],
                "Model": {
                    "MoE": "MoEPRS",
                    "MoE-global-int": "MoEPRS",
                    "MoE-CFG": "MultiPRS",
                }[model_name],
            }
        )

    unif_proba = (1.0 / len(dataset.prs_ids)) * np.ones(shape=pred_proba.shape)
    metrics.append(evaluate_preds(true_proba, unif_proba))
    metrics[-1].update(
        {
            "Heritability": config["heritability"],
            "Simulation Scenario": config["simulation_type"],
            "Model": "Uniform",
        }
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate gating model performance")
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

    dataset_paths = glob.glob(
        f"data/harmonized_data_simulations/sim_*/{args.analysis_id}/{args.biobank}/*_h0.*/test_data.pkl"
    )
    if len(dataset_paths) == 0:
        raise FileNotFoundError(
            "No simulation test datasets found for "
            f"analysis_id={args.analysis_id}, biobank={args.biobank}."
        )

    results = Parallel(n_jobs=args.jobs, backend="multiprocessing")(
        delayed(evaluate_dataset)(path) for path in dataset_paths
    )

    # Flatten results (since evaluate_dataset returns a list)
    predictive_perf = [item for sublist in results for item in sublist]

    predictive_perf = pd.DataFrame(predictive_perf)
    predictive_perf["Simulation Scenario"] = map_sim_scenario_names(
        predictive_perf["Simulation Scenario"]
    )

    predictive_perf.rename(columns={"Simulation Scenario": "Scenario"}, inplace=True)

    sns.set_context("paper", font_scale=2.25)

    g = sns.catplot(
        data=predictive_perf,
        col="Scenario",
        col_order=get_sim_order(predictive_perf["Scenario"].unique()),
        col_wrap=3,
        hue_order=["MoEPRS", "MultiPRS", "Uniform"],
        x="Heritability",
        y="Brier score",
        hue="Model",
        kind="bar",
        palette={"MoEPRS": "#375E97", "MultiPRS": "#FFBB00", "Uniform": "#d3d3d3"},
    )

    for ax in g.axes.flat:
        title = ax.get_title()
        if title.startswith("Scenario = "):
            ax.set_title(title.replace("Scenario = ", ""))

    plt.savefig(
        f"figures/simulations/gate_performance_Brier_{args.analysis_id}_{args.biobank}.eps"
    )
    plt.close()

    discrete_perf_subset = predictive_perf.loc[
        predictive_perf["Scenario"].isin(
            ["Single model", "Discrete context (Sex)", "Discrete context (Ancestry)"]
        )
    ]

    discrete_perf_subset = discrete_perf_subset.loc[
        discrete_perf_subset["Model"] != "Uniform"
    ]

    g = sns.catplot(
        data=discrete_perf_subset,
        col="Scenario",
        col_order=get_sim_order(discrete_perf_subset["Scenario"].unique()),
        col_wrap=3,
        hue_order=["MoEPRS", "MultiPRS"],
        x="Heritability",
        y="Accuracy",
        hue="Model",
        kind="bar",
        palette={"MoEPRS": "#375E97", "MultiPRS": "#FFBB00"},
    )
    for ax in g.axes.flat:
        title = ax.get_title()
        if title.startswith("Scenario = "):
            ax.set_title(title.replace("Scenario = ", ""))

    plt.savefig(
        f"figures/simulations/gate_performance_accuracy_{args.analysis_id}_{args.biobank}.eps"
    )
    plt.close()
