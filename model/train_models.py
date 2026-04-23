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
from moe_pytorch import Lit_MoEPRS, make_deterministic, train_model
from PRSDataset import PRSDataset


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
    # First the base models with covariates:
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

    analysis_to_table = get_analysis_to_table_mapper()
    analysis_table_id = analysis_to_table.get(dataset.analysis_id)
    use_prs_in_gate = args.add_prs_to_gate or (
        analysis_table_id == "multitrait_prs_table"
    )

    gate_input = list(dataset.covariates_cols)
    if use_prs_in_gate:
        gate_input += list(dataset.prs_cols)

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


def train_moe_models_torch(
    dataset,
    gate_model_layers=None,
    add_covariates_to_experts=False,
    loss="likelihood_mixture2",
    optimizer="Adam",
    gate_add_batch_norm=True,
    gate_add_layer_norm=False,
    weight_decay=0.0,
    learning_rate=1e-3,
    max_epochs=1000,
    batch_size=2048,
    weigh_samples=False,
    seed=8,
    topk_k=None,
    tau_start=1.5,
    tau_end=1.0,
    tau_warm_epochs=10,
    tau_decay_epochs=90,
    hard_ste=False,
    lb_coef=0.00,
    ent_coef=0.05,
    ent_coef_end=0.0,
    ent_warm_epochs=10,
    ent_decay_epochs=90,
    ancestry_balance_lambda=None,
    use_per_expert_bias=False,
    use_global_head=True,
    global_head_bias=True,
    fix_sigma2=False,
    binomial_logit_level=False,
):
    dataset.set_backend("torch")

    # -----------------------------------------
    # Gating model input:

    gate_input = list(dataset.covariates_cols)
    if args.add_prs_to_gate:
        gate_input += list(dataset.prs_cols)

    # -----------------------------------------

    # Define which columns to fetch
    group_getitem_cols = {
        "phenotype": [dataset.phenotype_col],
        "gate_input": gate_input,
        "experts": dataset.prs_cols,
        "global_input": dataset.covariates_cols,
    }

    # whether or not to have expert-specific covariates slopes
    if add_covariates_to_experts:
        group_getitem_cols["expert_covariates"] = dataset.covariates_cols

    dataset.set_group_getitem_cols(group_getitem_cols)

    # If gaussian, and sigma2 is estimated, use the corresponding sigma2 included loss
    loss = (
        "likelihood_mixture_sigma"
        if (dataset.phenotype_likelihood == "gaussian" and not fix_sigma2)
        else loss
    )

    # Initialize the SGD MoE model through pytorch Lightning:
    m = Lit_MoEPRS(
        dataset.group_getitem_cols,
        gate_model_layers=gate_model_layers,
        loss=loss,
        family=dataset.phenotype_likelihood,
        optimizer=optimizer,
        gate_add_batch_norm=gate_add_batch_norm,
        gate_add_layer_norm=gate_add_layer_norm,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        topk_k=topk_k,
        tau_start=tau_start,
        tau_end=tau_end,
        tau_warm_epochs=tau_warm_epochs,
        tau_decay_epochs=tau_decay_epochs,
        hard_ste=hard_ste,
        lb_coef=lb_coef,
        ent_coef=ent_coef,
        ent_coef_end=ent_coef_end,
        ent_warm_epochs=ent_warm_epochs,
        ent_decay_epochs=ent_decay_epochs,
        use_per_expert_bias=use_per_expert_bias,
        use_global_head=use_global_head,
        global_head_bias=global_head_bias,
        binomial_logit_level=binomial_logit_level,
    )

    # Train with PyTorch Lightning:
    with Timer() as t:
        _, m = train_model(
            m,
            dataset,
            max_epochs=max_epochs,
            batch_size=batch_size,
            weigh_samples=weigh_samples,
            seed=seed,
            split_seed=seed,
            ancestry_balance_lambda=ancestry_balance_lambda,
        )

    m.runtime_minutes = t.minutes  # <-- attach to model for saving later

    return m


def train_all_models(
    dataset,
    baseline_kwargs,
    moe_kwargs=None,
    skip_baseline=False,
    skip_moe=False,
    moe_pytorch_kwargs=None,
    pytorch_only=False,
    skip_moe_pytorch=False,
    seed=8,
):
    trained_models = {}
    runtimes = {}
    moe_kwargs = moe_kwargs or {}

    if not pytorch_only:
        if not skip_baseline:
            bm, br = train_baseline_linear_models(dataset, **baseline_kwargs)
            trained_models.update(bm)
            runtimes.update(br)

        if not skip_moe:
            mm, mr = train_moe_model_numpy(dataset, **moe_kwargs)
            trained_models.update(mm)
            runtimes.update(mr)

    if not skip_moe_pytorch:
        pt_kwargs = Lit_MoEPRS.default_training_kwargs(seed=seed)

        if moe_pytorch_kwargs:
            pt_kwargs.update(moe_pytorch_kwargs)

        m_pt = train_moe_models_torch(dataset, **pt_kwargs)

        trained_models["MoE-PyTorch"] = m_pt
        if hasattr(m_pt, "runtime_minutes"):
            runtimes["MoE-PyTorch"] = float(m_pt.runtime_minutes)

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
        "--moe-pytorch-kwargs",
        dest="moe_pytorch_kwargs",
        type=str,
        default="",
        help="Comma-separated key=value pairs for PyTorch MoE (e.g. max_epochs=500,learning_rate=1e-3,gate_add_layer_norm=True).",
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
        "--skip-moe-pytorch",
        dest="skip_moe_pytorch",
        action="store_true",
        default=False,
        help="Whether to skip training the MoE models with PyTorch.",
    )
    parser.add_argument(
        "--pytorch-only",
        dest="pytorch_only",
        action="store_true",
        default=False,
        help="Whether to only train the MoE models with PyTorch.",
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

    parser.add_argument(
        "--moe-pytorch-json",
        dest="moe_pytorch_json",
        type=str,
        default="",
        help="JSON string for PyTorch MoE kwargs (overrides --moe-pytorch-kwargs).",
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
    baseline_kwargs = {}
    if len(args.baseline_kwargs) > 0:
        baseline_kwargs = {
            k: v
            for k, v in [kw.split("=") for kw in args.baseline_kwargs.split(",")]
            if v
        }

    # Extract options for training MoE PyTorch:
    moe_pytorch_kwargs = {}
    if len(args.moe_pytorch_kwargs) > 0:
        moe_pytorch_kwargs = {
            k: v
            for k, v in [kw.split("=") for kw in args.moe_pytorch_kwargs.split(",")]
            if v
        }

    # ----------------------------------------------------
    # Train all models:
    trained_models, model_runtimes = train_all_models(
        prs_dataset,
        baseline_kwargs,
        skip_baseline=args.skip_baseline,
        skip_moe=args.skip_moe,
        pytorch_only=args.pytorch_only,
        skip_moe_pytorch=args.skip_moe_pytorch,
        moe_pytorch_kwargs=moe_pytorch_kwargs,
        seed=args.seed,
    )

    # ----------------------------------------------------
    # Save the trained models (and associated statistics):

    print("> Saving trained models to:\n\t", output_dir)

    makedir(output_dir)

    for model_name, model in trained_models.items():
        runtime_min = model_runtimes.get(model_name, None)

        out = osp.join(output_dir, f"{model_name}{getattr(model, 'save_ext', '.pkl')}")
        model.save(out)
        print(f"Saved model: {model_name}")

        if runtime_min is not None:
            payload = {"Runtime_min": runtime_min}
            with open(osp.join(output_dir, f"{model_name}_runtime.json"), "w") as f:
                json.dump(payload, f)
