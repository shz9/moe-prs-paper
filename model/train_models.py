import argparse
import json
import os.path as osp
import sys
from functools import partial

from magenpy.utils.system_utils import makedir

sys.path.append(osp.dirname(osp.dirname(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))

from baseline_models import AncestryWeightedPRS, AttributePartitionedPRS, MultiPRS
from grid_search import custom_cv_grid_search, get_gate_penalty_ladder
from model_utils import Timer, get_analysis_to_table_mapper, get_model_name_mapper
from moe import MoEPRS
from moe_pytorch import TorchMoEPRS, make_deterministic
from PRSDataset import PRSDataset


def extract_kw_pairs(kw_string):
    if kw_string is None or len(kw_string) == 0:
        return {}
    else:
        return {k: v for k, v in [kw.split("=") for kw in kw_string.split(",")] if v}


def train_baseline_linear_models(
    dataset, penalty_type=None, penalty=0.0, class_weights=None, add_intercept=True
):
    dataset.set_backend("numpy")

    print(
        f"> Training baseline models for {dataset.phenotype_col} with {dataset.N} samples..."
    )

    base_models = dict()
    runtimes = dict()

    base_models["MultiPRS"] = MultiPRS(
        prs_dataset=dataset,
        expert_cols=dataset.prs_cols,
        covariates_cols=dataset.covariates_cols,
        add_intercept=add_intercept,
        class_weights=class_weights,
        penalty_type=penalty_type,
        penalty=penalty,
    )

    with Timer() as timer:
        base_models["MultiPRS"].fit()

    runtimes["MultiPRS"] = timer.minutes

    # -------------------------------------------------
    # Determine if we should run the AncestryWeightedPRS model:

    analysis_to_table = get_analysis_to_table_mapper()
    analysis_table_id = analysis_to_table.get(dataset.analysis_id)

    if analysis_table_id == "multi_ancestry_prs_table":
        # First, map the model names to their corresponding training cohort:
        MODEL_NAME_MAP = get_model_name_mapper()
        model_names = {
            m: MODEL_NAME_MAP.get(dataset.analysis_id, {}).get(m, m)
            for m in dataset.prs_cols
        }

        # If we have at least two models whose names overlaps with
        # ancestry labels in the dataset, then run the AncestryWeightedPRS model:
        if (
            len(
                set(model_names.values()).intersection(
                    set(dataset.data["Ancestry"].unique())
                )
            )
            >= 2
        ):
            base_models["AncestryWeightedPRS"] = AncestryWeightedPRS(
                prs_dataset=dataset,
                expert_cols=dataset.prs_cols,
                covariates_cols=dataset.covariates_cols,
                add_intercept=add_intercept,
                class_weights=class_weights,
                penalty_type=penalty_type,
                penalty=penalty,
                expert_ancestry_map=model_names,
            )

            with Timer() as timer:
                base_models["AncestryWeightedPRS"].fit()

            runtimes["AncestryWeightedPRS"] = timer.minutes

    # -------------------------------------------------
    # Determine if we should run the sex-matched PRS model:

    if analysis_table_id == "sex_biased_prs_table":
        attribute_to_score_map = {}

        for pid in dataset.prs_cols:
            if pid.endswith("_F"):
                attribute_to_score_map[0.0] = pid
            elif pid.endswith("_M"):
                attribute_to_score_map[1.0] = pid

        base_models["SexMatchedPRS"] = AttributePartitionedPRS(
            prs_dataset=dataset,
            partition_attribute="Sex",
            attribute_to_score_map=attribute_to_score_map,
            covariates_cols=dataset.covariates_cols,
            add_intercept=add_intercept,
            class_weights=class_weights,
            penalty_type=penalty_type,
            penalty=penalty,
        )

        with Timer() as timer:
            base_models["SexMatchedPRS"].fit()

        runtimes["SexMatchedPRS"] = timer.minutes

    # -------------------------------------------------
    # The base models with covariates:
    for i, pgs_id in enumerate(dataset.prs_cols):
        base_models[f"{pgs_id}-covariates"] = MultiPRS(
            prs_dataset=dataset,
            expert_cols=pgs_id,
            covariates_cols=dataset.covariates_cols,
            add_intercept=add_intercept,
            class_weights=class_weights,
            penalty_type=penalty_type,
            penalty=penalty,
        )

        with Timer() as timer:
            base_models[f"{pgs_id}-covariates"].fit()

        runtimes[f"{pgs_id}-covariates"] = timer.minutes

    # -------------------------------------------------
    # The model with only covariates:

    base_models["Covariates"] = MultiPRS(
        prs_dataset=dataset,
        covariates_cols=dataset.covariates_cols,
        add_intercept=add_intercept,
        class_weights=class_weights,
        penalty_type=penalty_type,
        penalty=penalty,
    )

    with Timer() as timer:
        base_models["Covariates"].fit()

    runtimes["Covariates"] = timer.minutes

    return base_models, runtimes


def train_moe_model_numpy(dataset):
    print(
        f"> Training MoE model for {dataset.phenotype_col} with {dataset.N} samples..."
    )

    dataset.set_backend("numpy")

    moe_models = dict()
    runtimes = dict()

    # -----------------------------------------
    # Gating model input:

    gate_input = list(dataset.covariates_cols)
    prs_gate_input = gate_input + list(dataset.prs_cols)

    # -----------------------------------------
    # Fit the standard MoEPRS model:
    moe = MoEPRS(
        prs_dataset=dataset,
        expert_cols=dataset.prs_cols,
        gate_input_cols=gate_input,
        global_covariates_cols=dataset.covariates_cols,
    )

    print("------------------------------------")
    print("> Training standard MoE model...")
    with Timer() as timer:
        moe.fit()

    moe_models["MoE"] = moe
    runtimes["MoE"] = timer.minutes

    print("------------------------------------")

    if args.add_prs_to_gate:
        print("------------------------------------")
        print("> Training standard MoE model with PRS in the gate...")

        moe_prs_gate = MoEPRS(
            prs_dataset=dataset,
            expert_cols=dataset.prs_cols,
            gate_input_cols=prs_gate_input,
            global_covariates_cols=dataset.covariates_cols,
        )

        with Timer() as timer:
            moe_prs_gate.fit()

        moe_models["MoE-prs-gating"] = moe_prs_gate
        runtimes["MoE-prs-gating"] = timer.minutes

        print("------------------------------------")

    # -----------------------------------------
    # Fit MoEPRS model covariate-free gating (e.g. MultiPRS)
    moe_cfg = MoEPRS(
        prs_dataset=dataset,
        expert_cols=dataset.prs_cols,
        gate_input_cols=None,
        global_covariates_cols=dataset.covariates_cols,
    )

    print("------------------------------------")
    print("> Training MoE model no input covariates to the gate...")

    with Timer() as timer:
        moe_cfg.fit()

    moe_models["MoE-CFG"] = moe_cfg
    runtimes["MoE-CFG"] = timer.minutes

    print("------------------------------------")

    # -----------------------------------------
    # Fit the MoEPRS model using grid search:

    print("------------------------------------")
    print("> Training MoE model with grid search...")

    partial_moe = partial(
        MoEPRS,
        expert_cols=dataset.prs_cols,
        gate_input_cols=gate_input,
        global_covariates_cols=dataset.covariates_cols,
    )

    with Timer() as timer:
        moe_models["MoE-GS"] = custom_cv_grid_search(
            dataset,
            partial_moe,
            {"gate_penalty": get_gate_penalty_ladder()},
            n_jobs=4,
            validation_fit_params={"verbose": False, "n_iter": 100},
        )

    runtimes["MoE-GS"] = timer.minutes

    print("------------------------------------")

    if args.add_prs_to_gate:
        print("------------------------------------")
        print("> Training MoE model with PRS in the gate and grid search...")

        partial_moe_prs_gate = partial(
            MoEPRS,
            expert_cols=dataset.prs_cols,
            gate_input_cols=prs_gate_input,
            global_covariates_cols=dataset.covariates_cols,
        )

        with Timer() as timer:
            moe_models["MoE-GS-prs-gating"] = custom_cv_grid_search(
                dataset,
                partial_moe_prs_gate,
                {"gate_penalty": get_gate_penalty_ladder()},
                n_jobs=4,
                validation_fit_params={"verbose": False, "n_iter": 100},
            )

        runtimes["MoE-GS-prs-gating"] = timer.minutes

        print("------------------------------------")

    # -----------------------------------------
    # Run MoEPRS with fixed residuals:

    if dataset.phenotype_likelihood != "binomial":
        print("------------------------------------")
        print("> Training MoE model with fixed residuals...")

        moe_fix_resid = MoEPRS(
            prs_dataset=dataset,
            expert_cols=dataset.prs_cols,
            gate_input_cols=gate_input,
            global_covariates_cols=dataset.covariates_cols,
            fix_residuals=True,
        )

        with Timer() as timer:
            moe_fix_resid.fit()

        runtimes["MoE-fixed-resid"] = timer.minutes
        moe_models["MoE-fixed-resid"] = moe_fix_resid
        print("------------------------------------")

    return moe_models, runtimes


def train_moe_models_torch(dataset, **kwargs):
    print(
        f"> Training TorchMoEPRS model for {dataset.phenotype_col} with {dataset.N} samples..."
    )
    dataset.set_backend("torch")

    gate_input = list(dataset.covariates_cols)
    prs_gate_input = gate_input + list(dataset.prs_cols)

    moe_models = dict()
    runtimes = dict()

    fit_keys = {
        "min_epochs",
        "max_epochs",
        "prop_validation",
        "min_validation",
        "batch_size",
        "weigh_samples",
        "seed",
        "ancestry_balance_lambda",
    }
    fit_kwargs = {k: kwargs[k] for k in fit_keys if k in kwargs}
    model_kwargs = {k: v for k, v in kwargs.items() if k not in fit_keys}

    def fit_torch_models(gate_cols, suffix=""):
        model = TorchMoEPRS(
            prs_dataset=dataset,
            expert_cols=dataset.prs_cols,
            gate_input_cols=gate_cols,
            global_covariates_cols=dataset.covariates_cols,
            **model_kwargs,
        )

        with Timer() as timer:
            model.fit(**fit_kwargs)

        moe_models[f"TorchMoEPRS{suffix}"] = model
        runtimes[f"TorchMoEPRS{suffix}"] = timer.minutes

        if dataset.phenotype_likelihood == "binomial":
            model_ens = TorchMoEPRS(
                prs_dataset=dataset,
                expert_cols=dataset.prs_cols,
                gate_input_cols=gate_cols,
                global_covariates_cols=dataset.covariates_cols,
                loss="ensemble_loss",
                binomial_mixing_level="logit",
                **model_kwargs,
            )

            with Timer() as timer:
                model_ens.fit(**fit_kwargs)
        else:
            model_ens = TorchMoEPRS(
                prs_dataset=dataset,
                expert_cols=dataset.prs_cols,
                gate_input_cols=gate_cols,
                global_covariates_cols=dataset.covariates_cols,
                loss="ensemble_loss",
                **model_kwargs,
            )

            with Timer() as timer:
                model_ens.fit(**fit_kwargs)

        moe_models[f"TorchMoEPRS-ensemble{suffix}"] = model_ens
        runtimes[f"TorchMoEPRS-ensemble{suffix}"] = timer.minutes

    fit_torch_models(gate_input)

    if args.add_prs_to_gate:
        fit_torch_models(prs_gate_input, suffix="-prs-gating")

    return moe_models, runtimes


def train_all_models(
    dataset,
    baseline_kwargs=None,
    moe_kwargs=None,
    moe_torch_kwargs=None,
    skip_baseline=False,
    skip_moe=False,
    skip_torch_moe=False,
    seed=8,
):

    trained_models = {}
    runtimes = {}

    baseline_kwargs = baseline_kwargs or {}
    moe_kwargs = moe_kwargs or {}
    moe_torch_kwargs = moe_torch_kwargs or {}

    if not skip_baseline:
        bm, br = train_baseline_linear_models(dataset, **baseline_kwargs)
        trained_models.update(bm)
        runtimes.update(br)

    if not skip_moe:
        mm, mr = train_moe_model_numpy(dataset, **moe_kwargs)
        trained_models.update(mm)
        runtimes.update(mr)

    if not skip_torch_moe:
        pm, pr = train_moe_models_torch(dataset, **moe_torch_kwargs)
        trained_models.update(pm)
        runtimes.update(pr)

    return trained_models, runtimes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline and MoE models.")
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        type=str,
        required=True,
        help="The path to the dataset file.",
    )
    parser.add_argument(
        "--baseline-kwargs",
        dest="baseline_kwargs",
        type=str,
        default="",
        help="A comma-separated list of key-value pairs with the arguments for the baseline models.",
    )
    parser.add_argument(
        "--moe-kwargs",
        dest="moe_kwargs",
        type=str,
        default="",
        help="A comma-separated list of key-value pairs with the arguments for the MoE model.",
    )
    parser.add_argument(
        "--moe-torch-kwargs",
        dest="moe_torch_kwargs",
        type=str,
        default="",
        help="Comma-separated key=value pairs for TorchMoEPRS (e.g. max_epochs=500,learning_rate=1e-3,gate_add_layer_norm=True).",
    )
    parser.add_argument(
        "--skip-baseline",
        dest="skip_baseline",
        action="store_true",
        default=False,
        help="Whether to skip training the baseline models.",
    )
    parser.add_argument(
        "--skip-moe",
        dest="skip_moe",
        action="store_true",
        default=False,
        help="Whether to skip training the MoE models.",
    )
    parser.add_argument(
        "--skip-torch-moe",
        dest="skip_torch_moe",
        action="store_true",
        default=False,
        help="Whether to skip training the MoE models with PyTorch.",
    )
    parser.add_argument(
        "--add-prs-to-gate",
        dest="add_prs_to_gate",
        action="store_true",
        default=False,
        help="If True, add PRS data as input to the gating model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=8,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    make_deterministic(args.seed)

    # Attempt to read the dataset object:
    prs_dataset = PRSDataset.from_pickle(args.dataset_path)

    # ----------------------------------------------------

    output_dir = osp.dirname(args.dataset_path).replace(
        "harmonized_data", "trained_models"
    )

    analysis_id = args.dataset_path.split("/")[2]
    dataset_name = osp.basename(args.dataset_path).replace(".pkl", "")
    output_dir = osp.join(output_dir, dataset_name)

    # ----------------------------------------------------
    # Extract some options for training baseline models:

    baseline_kwargs = extract_kw_pairs(args.baseline_kwargs)
    moe_kwargs = extract_kw_pairs(args.moe_kwargs)
    moe_torch_kwargs = extract_kw_pairs(args.moe_torch_kwargs)

    # ----------------------------------------------------
    # Train all models:
    trained_models, model_runtimes = train_all_models(
        prs_dataset,
        baseline_kwargs,
        moe_kwargs=moe_kwargs,
        moe_torch_kwargs=moe_torch_kwargs,
        skip_baseline=args.skip_baseline,
        skip_moe=args.skip_moe,
        skip_torch_moe=args.skip_torch_moe,
        seed=args.seed,
    )

    # ----------------------------------------------------
    # Save the trained models (and associated statistics):

    print("> Saving trained models to:\n\t", output_dir)

    makedir(output_dir)

    for model_name, model in trained_models.items():
        runtime_min = model_runtimes.get(model_name, None)

        out = osp.join(output_dir, f"{model_name}.pkl")
        model.save(out)
        print(f"Saved model: {model_name}")

        if runtime_min is not None:
            payload = {"Runtime_min": runtime_min}
            with open(osp.join(output_dir, f"{model_name}_runtime.json"), "w") as f:
                json.dump(payload, f)
