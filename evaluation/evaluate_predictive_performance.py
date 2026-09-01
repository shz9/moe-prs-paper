import argparse
import glob
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from magenpy.utils.system_utils import makedir
from viprs.eval.eval_utils import r2_stats

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))

from baseline_models import AncestryWeightedPRS, AttributePartitionedPRS, MultiPRS
from eval_utils import (
    BINARY_EVAL_METRICS,
    CONT_EVAL_METRICS,
    DEFAULT_BOOTSTRAP_CI,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_MIN_GROUP_SIZE,
    DEFAULT_MIN_CASES,
    EVAL_METRICS,
    INCREMENTAL_METRICS,
    bootstrap_metric_ci,
    generate_categorical_masks,
    generate_coarse_ancestry_masks,
    generate_continuous_masks,
    generate_pc_cluster_masks,
    generate_predictions,
    incremental_r2_from_predictions,
)
from moe import MoEPRS
from moe_pytorch import TorchMoEPRS
from PRSDataset import PRSDataset


def _resolve_metric_names(prs_dataset, metrics=None):
    if metrics is None:
        if prs_dataset.phenotype_likelihood == "gaussian":
            metrics = CONT_EVAL_METRICS
        else:
            metrics = BINARY_EVAL_METRICS

    if isinstance(metrics, str):
        metric_names = [metrics]
    elif isinstance(metrics, dict):
        metric_names = list(metrics.keys())
    else:
        metric_names = list(metrics)

    for m in metric_names:
        assert m in EVAL_METRICS, f"Metric {m} is not recognized."

    return metric_names


def _parse_model_id(model_id):
    left, sep, right = model_id.partition(":")
    if sep == "":
        train_biobank = None
        train_fold = None
        train_source = None
        model_name = model_id
    else:
        parts = left.split("/")
        train_biobank = parts[0] if len(parts) > 0 else None
        if len(parts) > 1 and parts[1].startswith("fold_"):
            train_fold = parts[1]
            train_source = "/".join(parts[2:]) if len(parts) > 2 else None
        else:
            train_fold = None
            train_source = "/".join(parts[1:]) if len(parts) > 1 else None
        model_name = right

    prediction_type = "prs_only" if model_name.endswith("-PRS-only") else "full"
    base_name = model_name.replace("-PRS-only", "")

    if "moe" in base_name.lower():
        model_category = "MoE"
    elif base_name == "MultiPRS":
        model_category = "MultiPRS"
    elif base_name == "AncestryWeightedPRS":
        model_category = "AncestryWeightedPRS"
    elif base_name == "SexMatchedPRS":
        model_category = "AttributePartitionedPRS"
    elif base_name == "Covariates":
        model_category = "Covariates"
    elif base_name.endswith("-covariates"):
        model_category = "SinglePRS+Covariates"
    else:
        model_category = "SinglePRS"

    return {
        "model_id": model_id,
        "model_name": base_name,
        "prediction_type": prediction_type,
        "model_category": model_category,
        "train_biobank": train_biobank,
        "train_fold": train_fold,
        "train_source": train_source,
    }


def _canonicalize_fold_model_id(model_id):
    """Remove the internal fold component from a batched model identifier."""
    if not isinstance(model_id, str):
        return model_id
    left, sep, right = model_id.partition(":")
    if sep == "":
        return model_id
    parts = left.split("/")
    if len(parts) > 1 and parts[1].startswith("fold_"):
        parts.pop(1)
    return "/".join(parts) + ":" + right


def average_fold_predictions(prs_dataset, trained_models):
    """Average individual-level predictions across matching fold models.

    Model identifiers are canonicalized by removing their ``fold_i`` path
    component. Full and PRS-only predictions remain separate model entries.
    Every retained model must have predictions from every represented fold.
    """
    if trained_models is None or len(trained_models) == 0:
        raise ValueError("At least one trained fold model is required.")

    fold_predictions = generate_predictions(prs_dataset, trained_models)
    if fold_predictions.empty:
        raise ValueError("None of the fold models generated predictions.")

    expected_folds = {
        meta["train_fold"]
        for meta in (_parse_model_id(model_id) for model_id in trained_models)
        if meta["train_fold"] is not None
    }
    if len(expected_folds) < 2:
        raise ValueError(
            "Fold-averaged predictions require models from at least two folds."
        )

    grouped_columns = {}
    grouped_folds = {}
    for model_id in fold_predictions.columns:
        canonical_id = _canonicalize_fold_model_id(model_id)
        grouped_columns.setdefault(canonical_id, []).append(model_id)
        grouped_folds.setdefault(canonical_id, set()).add(
            _parse_model_id(model_id)["train_fold"]
        )

    incomplete = {
        model_id: sorted(expected_folds - grouped_folds[model_id])
        for model_id in grouped_columns
        if grouped_folds[model_id] != expected_folds
    }
    if incomplete:
        raise ValueError(
            "Cannot average incomplete fold predictions. Missing folds by model: "
            f"{incomplete}"
        )

    ensemble_predictions = pd.DataFrame(
        {
            model_id: fold_predictions[model_columns].mean(axis=1)
            for model_id, model_columns in grouped_columns.items()
        },
        index=fold_predictions.index,
    )
    ensemble_catalog = pd.DataFrame(
        [_parse_model_id(model_id) for model_id in ensemble_predictions.columns]
    )
    ensemble_catalog["n_model_folds"] = len(expected_folds)
    return ensemble_predictions, ensemble_catalog


def _split_batched_external_metrics(eval_df, fold_names):
    """Split one batched evaluation table into canonical per-fold tables."""
    fold_tables = {}
    base_model_mask = eval_df["train_fold"].isna()

    for fold in fold_names:
        fold_df = eval_df.loc[
            base_model_mask | eval_df["train_fold"].eq(fold)
        ].copy()
        fold_df.loc[fold_df["train_fold"].isna(), "train_fold"] = fold
        fold_df["test_fold"] = fold

        for id_col in ("model_id", "ref_model_id"):
            if id_col in fold_df.columns:
                fold_df[id_col] = fold_df[id_col].map(
                    _canonicalize_fold_model_id
                )
        fold_tables[fold] = fold_df

    return fold_tables


def _resolve_reference_model_id(
    model_catalog,
    train_biobank,
    train_fold=None,
    ref_model_name="Covariates",
):
    """
    Resolve the model_id of the reference model (e.g. "Covariates") that was
    trained on the same biobank (`train_biobank`) as the model currently being
    evaluated. Returns None if no such reference model can be found (e.g. when
    `train_biobank` is None/unknown, or no matching entry exists in the catalog).
    """
    if train_biobank is None:
        return None

    mdf = model_catalog
    mdf = mdf[
        (mdf["model_name"] == ref_model_name)
        & (mdf["prediction_type"] == "full")
        & (mdf["train_biobank"] == train_biobank)
    ]
    if train_fold is not None and "train_fold" in mdf.columns:
        mdf = mdf.loc[mdf["train_fold"] == train_fold]
    if len(mdf) == 0:
        return None
    return mdf.iloc[0]["model_id"]


def _resolve_reference_model_ids(
    model_catalog,
    train_biobank,
    train_fold=None,
    test_biobank=None,
    ref_model_name="Covariates",
):
    """
    Resolve distinct covariate reference models for incremental metrics.
    Each returned tuple is (ref_model_biobank, ref_model_id).
    """
    refs = []

    train_ref = _resolve_reference_model_id(
        model_catalog,
        train_biobank,
        train_fold=train_fold,
        ref_model_name=ref_model_name,
    )
    if train_ref is not None:
        refs.append((train_biobank, train_ref))

    test_ref = _resolve_reference_model_id(
        model_catalog,
        test_biobank,
        train_fold=train_fold,
        ref_model_name=ref_model_name,
    )
    if test_ref is not None:
        refs.append((test_biobank, test_ref))

    seen = set()
    out = []
    for ref_model_biobank, ref_model_id in refs:
        key = (ref_model_biobank, ref_model_id)
        if key in seen:
            continue
        seen.add(key)
        out.append((ref_model_biobank, ref_model_id))

    return out


def stratified_evaluation(
    prs_dataset,
    trained_models=None,
    model_catalog=None,
    test_biobank=None,
    ref_model_name="Covariates",
    cat_group_cols=None,
    cont_group_cols=None,
    cont_group_bins=None,
    include_coarse_ancestry=False,
    coarse_ancestry_only=False,
    pc_clusters=None,
    metrics=None,
    evaluate_base_models=True,
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
    trained_predictions=None,
    bootstrap=False,
    n_bootstrap=DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_ci=DEFAULT_BOOTSTRAP_CI,
    random_state=None,
):
    if cont_group_cols is not None:
        assert cont_group_bins is not None, (
            "Bins must be provided for continuous group columns!"
        )

    if coarse_ancestry_only:
        include_coarse_ancestry = True
        if cat_group_cols is not None:
            if isinstance(cat_group_cols, str):
                cat_group_cols = [cat_group_cols]
            cat_group_cols = [
                col for col in cat_group_cols if col.lower() != "ancestry"
            ]
            if len(cat_group_cols) == 0:
                cat_group_cols = None

    prs_dataset.set_backend("numpy")

    if trained_models is not None and trained_predictions is not None:
        raise ValueError("Use either trained_models or trained_predictions, not both.")

    if trained_predictions is not None:
        preds = trained_predictions.reset_index(drop=True)
        if len(preds) != prs_dataset.N:
            raise ValueError(
                "trained_predictions must contain one row per dataset sample."
            )
    elif trained_models is None or len(trained_models) == 0:
        preds = None
    else:
        preds = generate_predictions(prs_dataset, trained_models)

    # Generate sample masks to stratify the dataset:
    msks = {}
    if cat_group_cols is not None:
        msks.update(
            generate_categorical_masks(prs_dataset, cat_group_cols, min_group_size)
        )
    if cont_group_cols is not None:
        msks.update(
            generate_continuous_masks(prs_dataset, cont_group_cols, cont_group_bins)
        )
    if pc_clusters is not None:
        msks.update(generate_pc_cluster_masks(prs_dataset, "median", pc_clusters))
    if include_coarse_ancestry:
        msks.update(generate_coarse_ancestry_masks(prs_dataset))

    dfs = []

    # Evaluate the models across everyone:

    edf = evaluate_prs_models(
        prs_dataset,
        trained_models=preds,
        model_catalog=model_catalog,
        test_biobank=test_biobank,
        ref_model_name=ref_model_name,
        metrics=metrics,
        evaluate_base_models=evaluate_base_models,
        eval_category="All",
        eval_group="All",
        bootstrap=bootstrap,
        n_bootstrap=n_bootstrap,
        bootstrap_ci=bootstrap_ci,
        random_state=random_state,
    )

    dfs.append(edf)

    for mg, msk_group in msks.items():
        print("> Evaluation group:", mg)
        for m, msk in msk_group.items():
            print("\t> Subgroup:", m)

            try:
                edf = evaluate_prs_models(
                    prs_dataset,
                    trained_models=preds,
                    model_catalog=model_catalog,
                    test_biobank=test_biobank,
                    ref_model_name=ref_model_name,
                    mask=msk,
                    min_group_size=min_group_size,
                    metrics=metrics,
                    evaluate_base_models=evaluate_base_models,
                    eval_category=mg,
                    eval_group=m,
                    bootstrap=bootstrap,
                    n_bootstrap=n_bootstrap,
                    bootstrap_ci=bootstrap_ci,
                    random_state=random_state,
                )
            except Exception as e:
                continue

            if edf is None:
                continue

            dfs.append(edf)

    return pd.concat(dfs, ignore_index=True)


def evaluate_prs_models(
    prs_dataset,
    trained_models=None,
    model_catalog=None,
    test_biobank=None,
    ref_model_name="Covariates",
    mask=None,
    metrics=None,
    evaluate_base_models=True,
    min_group_size=DEFAULT_MIN_GROUP_SIZE,
    eval_category="All",
    eval_group="All",
    bootstrap=False,
    n_bootstrap=DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_ci=DEFAULT_BOOTSTRAP_CI,
    random_state=None,
):
    prs_dataset.set_backend("numpy")

    if mask is None:
        mask = np.ones(prs_dataset.N).astype(bool)

    if mask.sum() < min_group_size:
        raise ValueError(
            f"Skipping evaluation due to insufficient sample size ({mask.sum()} < {min_group_size})"
        )

    metric_names = _resolve_metric_names(prs_dataset, metrics=metrics)

    # --------------------------------------------------------------------------
    # Extract the phenotype:

    phenotype = prs_dataset.get_phenotype().flatten()[mask]

    # Sanity checks on the phenotype:
    if np.var(phenotype) == 0.0:
        raise ValueError("No phenotypic variance in this group of individuals!")

    if prs_dataset.phenotype_likelihood == "binomial":
        num_cases = int(phenotype.sum())
        if num_cases < DEFAULT_MIN_CASES or num_cases > phenotype.shape[0] - DEFAULT_MIN_CASES:
            raise ValueError(
                f"Highly unbalanced case/control numbers (Proportion: {num_cases}/{phenotype.shape[0]}); Cannot compute metrics reliably."
            )

    # --------------------------------------------------------------------------
    # Extract the polygenic scores to evaluate:
    if evaluate_base_models:
        prs_df = pd.DataFrame(
            prs_dataset.get_prs_predictions()[mask, :], columns=prs_dataset.prs_ids
        )
        base_meta = pd.DataFrame(
            [
                {
                    "model_id": m,
                    "model_name": m,
                    "prediction_type": "full",
                    "model_category": "SinglePRS",
                    "train_biobank": None,
                    "train_fold": None,
                    "train_source": None,
                }
                for m in prs_dataset.prs_ids
            ]
        )
    else:
        prs_df = None
        base_meta = pd.DataFrame(
            columns=[
                "model_id",
                "model_name",
                "prediction_type",
                "model_category",
                "train_biobank",
                "train_fold",
                "train_source",
            ]
        )

    if trained_models is not None:
        fitted_df = trained_models.loc[mask, :].reset_index(drop=True)
        if prs_df is None:
            prs_df = fitted_df
        else:
            prs_df = pd.concat([prs_df, fitted_df], axis=1)

    if prs_df is None:
        raise ValueError("No models to evaluate!")

    # --------------------------------------------------------------------------
    # Extract the covariates (if the metric requires them):

    if any([m in metric_names for m in INCREMENTAL_METRICS]):
        covar = pd.DataFrame(prs_dataset.get_covariates()[mask, :])

        # Remove invariant columns from the covariates:
        # Mainly relevant when evaluating on age groups or sex...
        covar = covar.loc[:, covar.var(axis=0) > 0]
    else:
        covar = None

    # --------------------------------------------------------------------------

    model_ids = list(prs_df.columns)
    meta_df = base_meta
    if model_catalog is not None and len(model_catalog) > 0:
        meta_df = pd.concat([meta_df, model_catalog], ignore_index=True)
    if len(meta_df) > 0:
        meta_df = meta_df.drop_duplicates(subset=["model_id"], keep="first")

    records = []

    for model_id in model_ids:
        # Keep only records with valid PGS values:
        keep = ~np.isnan(prs_df[model_id].values)
        n = int(keep.sum())
        if n < min_group_size:
            continue

        # Extract the phenotype values for the remaining samples:
        phenotype_values = phenotype[keep]

        # If the number cases/controls in the remaining samples is
        # too small, skip this model:
        if prs_dataset.phenotype_likelihood == "binomial":
            num_cases = int(phenotype_values.sum())
            if num_cases < 10 or num_cases > phenotype_values.shape[0] - 10:
                continue

        pred_values = prs_df[model_id].values[keep]

        row_meta = (
            meta_df[meta_df["model_id"] == model_id].iloc[0].to_dict()
            if len(meta_df[meta_df["model_id"] == model_id]) > 0
            else _parse_model_id(model_id)
        )

        # Resolve reference models (e.g. "Covariates") trained on:
        #   1. the same biobank as the model currently being evaluated
        #   2. the current test biobank
        # This is done fresh for every model_id, since different models may have
        # been trained on different biobanks.
        ref_model_ids = []
        if ref_model_name is not None:
            ref_model_ids = _resolve_reference_model_ids(
                meta_df,
                row_meta.get("train_biobank"),
                train_fold=row_meta.get("train_fold"),
                test_biobank=test_biobank,
                ref_model_name=ref_model_name,
            )

        for metric in metric_names:
            try:
                if bootstrap:
                    if metric in INCREMENTAL_METRICS:
                        metric_result = bootstrap_metric_ci(
                            phenotype_values,
                            pred_values,
                            metric_func=EVAL_METRICS[metric],
                            metric_args=(covar.loc[keep, :].to_numpy(),),
                            phenotype_likelihood=prs_dataset.phenotype_likelihood,
                            n_bootstrap=n_bootstrap,
                            ci=bootstrap_ci,
                            random_state=random_state,
                            min_samples=min_group_size,
                            min_cases=DEFAULT_MIN_CASES,
                        )
                    else:
                        metric_result = bootstrap_metric_ci(
                            phenotype_values,
                            pred_values,
                            metric=metric,
                            phenotype_likelihood=prs_dataset.phenotype_likelihood,
                            n_bootstrap=n_bootstrap,
                            ci=bootstrap_ci,
                            random_state=random_state,
                            min_samples=min_group_size,
                            min_cases=DEFAULT_MIN_CASES,
                        )
                    value = metric_result["value"]
                    se = metric_result["se"]
                elif metric in INCREMENTAL_METRICS:
                    value = EVAL_METRICS[metric](
                        phenotype_values, pred_values, covar.loc[keep, :]
                    )
                    metric_result = None
                else:
                    value = EVAL_METRICS[metric](phenotype_values, pred_values)
                    metric_result = None

                if not bootstrap and "R2" in metric:
                    try:
                        se = r2_stats(value, n)["SE"]
                    except AssertionError:
                        se = np.nan
                elif not bootstrap:
                    se = np.nan

            except Exception as e:
                print(f"Error evaluating metric {metric} on model {model_id}: {e}")
                value = np.nan
                se = np.nan
                metric_result = None

            records.append(
                {
                    **row_meta,
                    "metric": metric,
                    "metric_kind": "base",
                    "ref_model_id": None,
                    "ref_model_name": None,
                    "ref_model_biobank": None,
                    "value": value,
                    "se": se,
                    **(
                        {
                            "ci_lower": (
                                metric_result["ci_lower"]
                                if metric_result is not None
                                else np.nan
                            ),
                            "ci_upper": (
                                metric_result["ci_upper"]
                                if metric_result is not None
                                else np.nan
                            ),
                            "n_bootstrap": (
                                metric_result["n_bootstrap"]
                                if metric_result is not None
                                else n_bootstrap
                            ),
                            "n_bootstrap_valid": (
                                metric_result["n_bootstrap_valid"]
                                if metric_result is not None
                                else 0
                            ),
                            "uncertainty_method": "participant_bootstrap",
                        }
                        if bootstrap
                        else {}
                    ),
                    "n": metric_result["n"] if metric_result is not None else n,
                    "eval_category": eval_category,
                    "eval_group": eval_group,
                }
            )

            # Additional incremental layers vs reference models:
            if metric not in INCREMENTAL_METRICS:
                continue
            if row_meta.get("prediction_type", "full") != "full":
                continue

            for ref_model_biobank, ref_model_id in ref_model_ids:
                if ref_model_id not in model_ids or model_id == ref_model_id:
                    continue

                ref_keep = ~np.isnan(prs_df[ref_model_id].values)
                keep_both = keep & ref_keep
                n_both = int(keep_both.sum())
                if n_both >= min_group_size:
                    try:
                        if bootstrap:
                            delta_result = bootstrap_metric_ci(
                                phenotype[keep_both],
                                prs_df[model_id].values[keep_both],
                                metric_func=lambda y, full_pred, null_pred: (
                                    incremental_r2_from_predictions(
                                        y,
                                        full_pred,
                                        null_pred,
                                        metric=metric,
                                    )
                                ),
                                metric_args=(
                                    prs_df[ref_model_id].values[keep_both],
                                ),
                                phenotype_likelihood=prs_dataset.phenotype_likelihood,
                                n_bootstrap=n_bootstrap,
                                ci=bootstrap_ci,
                                random_state=random_state,
                                min_samples=min_group_size,
                                min_cases=DEFAULT_MIN_CASES,
                            )
                            delta_val = delta_result["value"]
                            delta_se = delta_result["se"]
                        else:
                            delta_result = None
                            delta_val = incremental_r2_from_predictions(
                                phenotype[keep_both],
                                prs_df[model_id].values[keep_both],
                                prs_df[ref_model_id].values[keep_both],
                                metric=metric,
                            )
                    except Exception as e:
                        continue
                    ref_meta = (
                        meta_df[meta_df["model_id"] == ref_model_id].iloc[0].to_dict()
                        if len(meta_df[meta_df["model_id"] == ref_model_id]) > 0
                        else _parse_model_id(ref_model_id)
                    )
                    if not bootstrap:
                        delta_se = np.nan
                    if not bootstrap and "R2" in metric:
                        try:
                            delta_se = r2_stats(delta_val, n_both)["SE"]
                        except AssertionError:
                            delta_se = 0.0
                    records.append(
                        {
                            **row_meta,
                            "metric": metric,
                            "metric_kind": "incremental_vs_ref",
                            "ref_model_id": ref_model_id,
                            "ref_model_name": ref_meta.get("model_name"),
                            "ref_model_biobank": ref_model_biobank,
                            "value": delta_val,
                            "se": delta_se,
                            **(
                                {
                                    "ci_lower": (
                                        delta_result["ci_lower"]
                                        if delta_result is not None
                                        else np.nan
                                    ),
                                    "ci_upper": (
                                        delta_result["ci_upper"]
                                        if delta_result is not None
                                        else np.nan
                                    ),
                                    "n_bootstrap": (
                                        delta_result["n_bootstrap"]
                                        if delta_result is not None
                                        else n_bootstrap
                                    ),
                                    "n_bootstrap_valid": (
                                        delta_result["n_bootstrap_valid"]
                                        if delta_result is not None
                                        else 0
                                    ),
                                    "uncertainty_method": "participant_bootstrap",
                                }
                                if bootstrap
                                else {}
                            ),
                            "n": (
                                delta_result["n"]
                                if delta_result is not None
                                else n_both
                            ),
                            "eval_category": eval_category,
                            "eval_group": eval_group,
                        }
                    )

    if len(records) == 0:
        return pd.DataFrame(
            columns=[
                "model_id",
                "model_name",
                "prediction_type",
                "model_category",
                "train_biobank",
                "train_fold",
                "train_source",
                "metric",
                "metric_kind",
                "ref_model_id",
                "ref_model_name",
                "ref_model_biobank",
                "value",
                "se",
                "ci_lower",
                "ci_upper",
                "n_bootstrap",
                "n_bootstrap_valid",
                "uncertainty_method",
                "n",
                "eval_category",
                "eval_group",
            ]
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PRS models")

    parser.add_argument(
        "--test-data",
        dest="test_data",
        type=str,
        required=True,
        help=(
            "Path to a held-out fold or a full external dataset. Full external "
            "datasets also require --train-biobank and either --model-fold or "
            "--all-model-folds."
        ),
    )
    parser.add_argument(
        "--cat-group-cols",
        dest="cat_group_cols",
        type=str,
        nargs="+",
        default=None,
        help="The columns to use for categorical stratification.",
    )
    parser.add_argument(
        "--cont-group-cols",
        dest="cont_group_cols",
        type=str,
        nargs="+",
        default=None,
        help="The columns to use for continuous stratification.",
    )
    parser.add_argument(
        "--cont-group-bins",
        dest="cont_group_bins",
        type=int,
        nargs="+",
        default=None,
        help="The number of bins to use for continuous stratification.",
    )
    parser.add_argument(
        "--pc-clusters",
        dest="pc_clusters",
        type=int,
        default=None,
        help="The number of PC clusters to use for stratification.",
    )
    parser.add_argument(
        "--min-group-size",
        dest="min_group_size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        help="The minimum group size to consider for evaluation.",
    )
    parser.add_argument(
        "--model-source",
        dest="model_source",
        type=str,
        default=None,
        help="Optional substring filter for model directory names (e.g. 'subsampled' or 'train')",
    )
    parser.add_argument(
        "--model-fold",
        dest="model_fold",
        type=str,
        default=None,
        help=(
            "Fold of trained models to evaluate. This is inferred for a "
            "fold-specific test dataset. For full external data, use this or "
            "--all-model-folds."
        ),
    )
    parser.add_argument(
        "--all-model-folds",
        dest="all_model_folds",
        action="store_true",
        default=False,
        help=(
            "For a full external dataset, load and evaluate every available "
            "training fold in one process, average their individual-level "
            "predictions, and estimate uncertainty by bootstrapping the "
            "external participants."
        ),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        dest="bootstrap_resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="Number of participant bootstrap replicates for external ensembles.",
    )
    parser.add_argument(
        "--bootstrap-ci",
        dest="bootstrap_ci",
        type=float,
        default=DEFAULT_BOOTSTRAP_CI,
        help="Coverage of the participant-bootstrap confidence interval.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        dest="bootstrap_seed",
        type=int,
        default=42,
        help="Random seed for external participant bootstrapping.",
    )
    parser.add_argument(
        "--train-biobank",
        dest="train_biobank",
        type=str,
        default=None,
        help=(
            "Optionally restrict evaluation to models trained in one biobank. "
            "This is required when evaluating a full external dataset."
        ),
    )
    parser.add_argument(
        "--include-coarse-ancestry",
        dest="include_coarse_ancestry",
        action="store_true",
        default=False,
        help="Whether to include coarse ancestry groupings in the evaluation metrics.",
    )
    parser.add_argument(
        "--coarse-ancestry-only",
        dest="coarse_ancestry_only",
        action="store_true",
        default=False,
        help=(
            "Evaluate coarse EUR/non-EUR ancestry groups without evaluating "
            "detailed Ancestry labels. Other requested categories are retained."
        ),
    )
    parser.add_argument(
        "--trained-models-suffix",
        dest="trained_models_suffix",
        type=str,
        default="",
        help="Optional suffix used during training (loads from trained_models_<suffix>/...).",
    )
    parser.add_argument(
        "--ref-model-name",
        dest="ref_model_name",
        type=str,
        default="Covariates",
        help="Name of the reference model used for incremental metrics. For each evaluated "
        "model, the reference model with this name that was trained on the SAME biobank "
        "is used (resolved separately per model).",
    )

    args = parser.parse_args()

    print(f"Evaluating Meta PRS performance on {args.test_data}")

    prs_dataset = PRSDataset.from_pickle(args.test_data)

    # Obtain path for relevant trained models:

    trained_root = "trained_models"
    if args.trained_models_suffix and len(args.trained_models_suffix.strip()) > 0:
        trained_root = f"trained_models_{args.trained_models_suffix.strip()}"

    dataset_dir = osp.dirname(osp.normpath(args.test_data))
    dataset_parent = osp.basename(dataset_dir)
    is_fold_test = dataset_parent.startswith("fold_")
    batched_external_folds = False

    if is_fold_test:
        if args.all_model_folds:
            parser.error("--all-model-folds only applies to full external datasets.")
        test_fold = dataset_parent
        if args.model_fold is not None and args.model_fold != test_fold:
            parser.error(
                f"--model-fold={args.model_fold} does not match the test-data "
                f"directory ({test_fold})."
            )
        test_biobank = osp.basename(osp.dirname(dataset_dir))
        analysis_id = osp.basename(osp.dirname(osp.dirname(dataset_dir)))
        harmonized_analysis_root = osp.dirname(osp.dirname(dataset_dir))
        evaluation_scope = "held_out_fold"
    else:
        if args.model_fold is not None and args.all_model_folds:
            parser.error("Use either --model-fold or --all-model-folds, not both.")
        if args.model_fold is None and not args.all_model_folds:
            parser.error(
                "--model-fold or --all-model-folds is required when --test-data "
                "is a full external dataset."
            )
        if args.train_biobank is None:
            parser.error(
                "--train-biobank is required when --test-data is a full external "
                "dataset, to prevent loading models trained on that external cohort."
            )
        batched_external_folds = args.all_model_folds
        test_fold = "*" if batched_external_folds else args.model_fold
        test_biobank = osp.basename(dataset_dir)
        analysis_id = osp.basename(osp.dirname(dataset_dir))
        harmonized_analysis_root = osp.dirname(dataset_dir)
        evaluation_scope = "external_full"

    trained_models_root = harmonized_analysis_root.replace(
        "harmonized_data", trained_root
    )

    trained_models_path = osp.join(
        trained_models_root,
        args.train_biobank or "*",
        test_fold,
        args.model_source or "*",
        "*.pkl",
    )

    print(f"> Loading trained models from: {trained_models_path}")

    trained_models = {}
    model_catalog = []

    for f in glob.glob(trained_models_path):
        model_basename = osp.basename(f).replace(".pkl", "")
        model_path_parts = osp.relpath(f, trained_models_root).split(os.sep)
        if len(model_path_parts) != 4:
            raise ValueError(f"Unexpected trained model path: {f}")

        train_biobank, train_fold, train_source, _ = model_path_parts
        if not batched_external_folds and train_fold != test_fold:
            continue
        if args.train_biobank is not None and train_biobank != args.train_biobank:
            continue

        if batched_external_folds:
            model_prefix = (
                train_biobank + "/" + train_fold + "/" + train_source + ":"
            )
        else:
            model_prefix = train_biobank + "/" + train_source + ":"
        model_name = model_prefix + model_basename
        model_meta = _parse_model_id(model_name)
        model_meta["train_fold"] = train_fold
        model_catalog.append(model_meta)

        if "moe" in model_name.lower():
            if "torch" in model_name.lower():
                trained_models[model_name] = TorchMoEPRS.from_saved_model(f)
            else:
                trained_models[model_name] = MoEPRS.from_saved_model(f)
        elif "AncestryWeightedPRS" in model_name:
            trained_models[model_name] = AncestryWeightedPRS.from_saved_model(f)
        elif "SexMatchedPRS" in model_name:
            trained_models[model_name] = AttributePartitionedPRS.from_saved_model(f)
        else:
            trained_models[model_name] = MultiPRS.from_saved_model(f)

    if len(trained_models) == 0:
        raise FileNotFoundError(f"No trained models found in {trained_models_path}")

    model_catalog = pd.DataFrame(model_catalog).drop_duplicates(
        subset=["model_id"], keep="first"
    )

    if batched_external_folds:
        ensemble_predictions, ensemble_catalog = average_fold_predictions(
            prs_dataset, trained_models
        )
        eval_df = stratified_evaluation(
            prs_dataset,
            trained_predictions=ensemble_predictions,
            model_catalog=ensemble_catalog,
            test_biobank=test_biobank,
            ref_model_name=args.ref_model_name,
            cat_group_cols=args.cat_group_cols,
            cont_group_cols=args.cont_group_cols,
            cont_group_bins=args.cont_group_bins,
            include_coarse_ancestry=args.include_coarse_ancestry,
            coarse_ancestry_only=args.coarse_ancestry_only,
            pc_clusters=args.pc_clusters,
            min_group_size=args.min_group_size,
            evaluate_base_models=False,
            bootstrap=True,
            n_bootstrap=args.bootstrap_resamples,
            bootstrap_ci=args.bootstrap_ci,
            random_state=args.bootstrap_seed,
        )
    else:
        eval_df = stratified_evaluation(
            prs_dataset,
            trained_models,
            model_catalog=model_catalog,
            test_biobank=test_biobank,
            ref_model_name=args.ref_model_name,
            cat_group_cols=args.cat_group_cols,
            cont_group_cols=args.cont_group_cols,
            cont_group_bins=args.cont_group_bins,
            include_coarse_ancestry=args.include_coarse_ancestry,
            coarse_ancestry_only=args.coarse_ancestry_only,
            pc_clusters=args.pc_clusters,
            min_group_size=args.min_group_size,
        )

    # Canonical test metadata for downstream plotting/filtering.
    eval_df["analysis_id"] = analysis_id
    eval_df["test_biobank"] = test_biobank
    if batched_external_folds:
        eval_df["test_fold"] = "ensemble"
    else:
        eval_df["test_fold"] = test_fold
    eval_df["test_dataset"] = osp.basename(args.test_data).replace(".pkl", "")
    eval_df["evaluation_scope"] = (
        "external_ensemble_bootstrap"
        if batched_external_folds
        else evaluation_scope
    )

    eval_root = "evaluation"
    if args.trained_models_suffix and len(args.trained_models_suffix.strip()) > 0:
        eval_root = f"evaluation_{args.trained_models_suffix.strip()}"

    if batched_external_folds:
        output_path = args.test_data.replace("harmonized_data", eval_root).replace(
            ".pkl", ".csv"
        )
        print("> Saving bootstrapped ensemble metrics to:\n\t", output_path)
        makedir(os.path.dirname(output_path))
        eval_df.to_csv(output_path, index=False)
    elif is_fold_test:
        output_path = args.test_data.replace("harmonized_data", eval_root).replace(
            ".pkl", ".csv"
        )
        print("> Saving evaluation metrics to:\n\t", output_path)
        makedir(os.path.dirname(output_path))
        eval_df.to_csv(output_path, index=False)
    else:
        output_path = osp.join(
            dataset_dir.replace("harmonized_data", eval_root),
            test_fold,
            osp.basename(args.test_data).replace(".pkl", ".csv"),
        )
        print("> Saving evaluation metrics to:\n\t", output_path)
        makedir(os.path.dirname(output_path))
        eval_df.to_csv(output_path, index=False)
