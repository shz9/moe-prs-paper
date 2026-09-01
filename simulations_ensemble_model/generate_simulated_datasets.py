import copy
import os.path as osp
import sys

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
import argparse
import pickle

import numpy as np
from magenpy.utils.system_utils import makedir
from PRSDataset import PRSDataset
from sklearn.preprocessing import StandardScaler


def _parse_analysis_and_biobank(dataset_path, dataset):
    """
    Resolve the current analysis ID / biobank convention from a harmonized dataset path.

    Expected path convention:
        data/harmonized_data/{analysis_id}/{biobank}/{split}.pkl
    """

    path_parts = osp.normpath(dataset_path).split(osp.sep)

    try:
        harmonized_idx = path_parts.index("harmonized_data")
        path_analysis_id = path_parts[harmonized_idx + 1]
        biobank = path_parts[harmonized_idx + 2]
    except (ValueError, IndexError):
        path_analysis_id = None
        biobank = "unknown_biobank"

    analysis_id = getattr(dataset, "analysis_id", None) or path_analysis_id
    if analysis_id is None:
        raise ValueError(
            "Could not resolve analysis_id from the PRSDataset object or dataset path. "
            "Expected data/harmonized_data/{analysis_id}/{biobank}/{split}.pkl."
        )

    return analysis_id, biobank


def _add_residual_component(sim_pgs, h2):
    """
    Add a residual component to the simulated phenotype,
    taking into account the specified heritability.

    Args:
        sim_pgs (np.ndarray): Simulated PRS values.
        h2 (float): Heritability

    Returns:
        np.ndarray: Simulated phenotype values.
    """

    resid_var = np.var(sim_pgs, ddof=0) * (1 - h2) / (h2)
    return sim_pgs + np.sqrt(resid_var) * np.random.randn(*sim_pgs.shape)


def _output_result(dataset, simulated_phenotype, output_dir):
    """
    Output the harmonized dataset with the simulated phenotype.

    Args:
        dataset (PRSDataset): Dataset object.
        simulated_phenotype (np.ndarray): Simulated phenotype values.
        output_dir (str): Output directory path.
    """

    dataset_cp = copy.deepcopy(dataset)
    dataset_cp.data[dataset_cp.phenotype_col] = simulated_phenotype
    dataset_cp.save(osp.join(output_dir, "full_data.pkl"))

    train_data, test_data = dataset_cp.train_test_split(args.prop_test)

    train_data.save(osp.join(output_dir, "train_data.pkl"))
    test_data.save(osp.join(output_dir, "test_data.pkl"))


def single_model_simulation(dataset, h2, single_model_id=None):
    """
    Select a single model from the ensemble and simulate phenotypes
    based on the selected model's weights.
    """

    if single_model_id is None:
        # If not provided, select a random model ID
        single_model_id = np.random.choice(dataset.prs_ids)

    # Simulate the phenotype based on the selected model:
    pgs = dataset.get_data_columns(single_model_id)
    simulated_phenotypes = _add_residual_component(pgs, h2)

    # Output the simulated data:
    sim_output_dir = osp.join(global_output_dir, f"single_model_h{args.h2}/")

    makedir(sim_output_dir)

    _output_result(dataset, simulated_phenotypes, sim_output_dir)

    # Output the configuration data:
    with open(osp.join(sim_output_dir, "config.pkl"), "wb") as f:
        pickle.dump(
            {
                "simulation_type": "single_model",
                "single_model_id": single_model_id,
                "heritability": args.h2,
            },
            f,
        )


def multiprs_simulation(dataset, h2):
    """
    Simulate phenotypes using fixed mixing weights according
    to the MultiPRS model.
    """

    # Extract the polygenic scores:
    pgs = dataset.get_prs_predictions()

    # Standardize the polygenic scores:
    pgs = (pgs - pgs.mean()) / pgs.std()

    # Generate global mixing weights:
    mixing_weights = np.random.dirichlet(np.ones(pgs.shape[1]), size=1)[0]

    pgs = (pgs * mixing_weights).sum(axis=1)

    # Simulate the phenotype:
    simulated_phenotypes = _add_residual_component(pgs, h2)

    # Output the simulated data:
    sim_output_dir = osp.join(global_output_dir, f"multiprs_h{args.h2}/")

    makedir(sim_output_dir)

    _output_result(dataset, simulated_phenotypes, sim_output_dir)

    # Output the configuration data:
    with open(osp.join(sim_output_dir, "config.pkl"), "wb") as f:
        pickle.dump(
            {
                "simulation_type": "multiprs",
                "mixing_weights": mixing_weights,
                "heritability": args.h2,
            },
            f,
        )


def discrete_context_simulation(dataset, h2, context="Ancestry"):
    """
    Simulate phenotypes by assigning individuals from
    the same ancestry or sex a corresponding PRS.
    """

    assert context in ["Ancestry", "Sex"], "Invalid context"
    assert context in dataset.data.columns, "Invalid context"

    # Get the polygenic scores and standardize them:
    pgs = dataset.get_prs_predictions()
    pgs = (pgs - pgs.mean()) / pgs.std()

    # Get unique contexts:
    unique_contexts = dataset.data[context].unique()

    # Ensure that the number of unique contexts is greater than 1
    # and less than 20 (heuristic upper bound):
    assert len(unique_contexts) > 1 and len(unique_contexts) < 20, (
        "Number of unique contexts must be greater than 1 and less than 20"
    )

    # For each unique context, assign a single polygenic score randomly:
    selected_scores = np.random.choice(
        pgs.shape[1], size=len(unique_contexts), replace=context == "Ancestry"
    )
    context_to_scores = dict(zip(unique_contexts, selected_scores))

    context_idx = dataset.data[context].map(context_to_scores)

    # Generate the final polygenic scores based on the context selection:
    pgs = pgs[np.arange(pgs.shape[0]), context_idx]

    # Simulate the phenotype:
    simulated_phenotypes = _add_residual_component(pgs, h2)

    # Output the simulated data:
    sim_output_dir = osp.join(global_output_dir, f"context_{context}_h{args.h2}/")

    makedir(sim_output_dir)

    _output_result(dataset, simulated_phenotypes, sim_output_dir)

    # Output the configuration data:
    with open(osp.join(sim_output_dir, "config.pkl"), "wb") as f:
        pickle.dump(
            {
                "simulation_type": f"discrete_context ({context})",
                "context": context,
                "context_to_scores": context_to_scores,
                "heritability": args.h2,
            },
            f,
        )


def continuous_context_simulation(dataset, h2, context="Age"):
    """
    Simulate phenotypes by continuously mixing two PRSs,
    where the mixing weights are conditional on the
    specified context.
    """

    assert context in dataset.data.columns, "Invalid context"

    # Get the polygenic scores and standardize them:
    pgs = dataset.get_prs_predictions()
    pgs = (pgs - pgs.mean()) / pgs.std()

    # Get the context variable and standardize it:
    ctx = dataset.data[context].values
    ctx_stats = {"mean": ctx.mean(), "std": ctx.std()}
    ctx = (ctx - ctx_stats["mean"]) / ctx_stats["std"]

    # Select two random polygenic scores:
    pgs_idx1, pgs_idx2 = np.random.choice(pgs.shape[1], size=2, replace=False)
    # Randomly draw the weights for the weighing mechanism:
    w0, w1 = np.random.normal(size=2)
    # Determine the mixing weight for each individual:
    from scipy.special import expit

    w = expit(w0 + w1 * ctx)

    # Generate the final polygenic scores by mixing the two models:
    pgs = w * pgs[:, pgs_idx1] + (1 - w) * pgs[:, pgs_idx2]

    # Simulate the phenotype:
    simulated_phenotypes = _add_residual_component(pgs, h2)

    # Output the simulated data:
    sim_output_dir = osp.join(global_output_dir, f"context_{context}_h{args.h2}/")

    makedir(sim_output_dir)

    _output_result(dataset, simulated_phenotypes, sim_output_dir)

    # Output the configuration data:
    with open(osp.join(sim_output_dir, "config.pkl"), "wb") as f:
        pickle.dump(
            {
                "simulation_type": f"continuous_context ({context})",
                "context": context,
                "context_statistics": ctx_stats,
                "selected_scores": [pgs_idx1, pgs_idx2],
                "sigmoid_weights": [w0, w1],
                "heritability": args.h2,
            },
            f,
        )


def moe_simulation(dataset, h2):
    """
    Simulate phenotypes using a mixture of PRS models.
    """

    # Get the polygenic scores and standardize them:
    pgs = dataset.get_prs_predictions()
    pgs = (pgs - pgs.mean()) / pgs.std()

    # Define scaler for the covariates:
    covar_scaler = StandardScaler()

    # Get the covariates and standardize them:
    covars = dataset.get_covariates()
    covars = covar_scaler.fit_transform(covars)
    # Add intercept term
    covars = np.hstack([np.ones((covars.shape[0], 1)), covars])

    # Generate random gating weights:
    gate_weights = np.random.normal(size=(covars.shape[1], pgs.shape[1]))

    # Generate the mixing weights for each individual:
    from scipy.special import softmax

    mixing_weights = softmax(covars.dot(gate_weights), axis=1)

    pgs = (pgs * mixing_weights).sum(axis=1)

    # Simulate the phenotype:
    simulated_phenotypes = _add_residual_component(pgs, h2)

    # Output the simulated data:
    sim_output_dir = osp.join(global_output_dir, f"moe_h{args.h2}/")

    makedir(sim_output_dir)

    _output_result(dataset, simulated_phenotypes, sim_output_dir)

    # Output the configuration data:
    with open(osp.join(sim_output_dir, "config.pkl"), "wb") as f:
        pickle.dump(
            {
                "simulation_type": "moe",
                "gate_weights": gate_weights,
                "covar_scaler": covar_scaler,
                "heritability": args.h2,
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ensemble PRS simulations")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/harmonized_data/HEIGHT_MA/ukbb/full_data.pkl",
        help="Path to PRS dataset used for generating simulations.",
    )
    parser.add_argument(
        "--h2", type=float, default=0.6, help="Heritability of simulated phenotype."
    )
    parser.add_argument(
        "--prop-test",
        dest="prop_test",
        type=float,
        default=0.3,
        help="The proportion of samples to use for testing (default: 0.3).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/harmonized_data_simulations/",
        help="Output directory",
    )

    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Heritability:", args.h2)

    dataset = PRSDataset.from_pickle(args.dataset)
    analysis_id, biobank = _parse_analysis_and_biobank(args.dataset, dataset)
    global_output_dir = osp.join(args.output_dir, analysis_id, biobank)

    print("Analysis ID:", analysis_id)
    print("Biobank:", biobank)

    print("Simulating single model scenario...")
    single_model_simulation(dataset, args.h2)
    print("Simulating MultiPRS scenario...")
    multiprs_simulation(dataset, args.h2)
    print("Simulating discrete context (Ancestry) scenario...")
    discrete_context_simulation(dataset, args.h2, "Ancestry")
    print("Simulating discrete context (Sex) scenario...")
    discrete_context_simulation(dataset, args.h2, "Sex")
    print("Simulating discrete context (Age) scenario...")
    continuous_context_simulation(dataset, args.h2, "Age")
    print("Simulating MoE scenario...")
    moe_simulation(dataset, args.h2)
