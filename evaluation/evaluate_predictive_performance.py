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
    EVAL_METRICS,
    INCREMENTAL_METRICS,
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
        train_source = None
        model_name = model_id
    else:
        parts = left.split("/")
        train_biobank = parts[0] if len(parts) > 0 else None
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
        "train_source": train_source,
    }


def _resolve_reference_model_id(
    model_catalog, test_biobank, ref_model_name="Covariates"
):
    mdf = model_catalog
    mdf = mdf[
        (mdf["model_name"] == ref_model_name)
        & (mdf["prediction_type"] == "full")
        & (mdf["train_biobank"] == test_biobank)
    ]
    if len(mdf) == 0:
        return None
    return mdf.iloc[0]["model_id"]


def stratified_evaluation(
    prs_dataset,
    trained_models=None,
    model_catalog=None,
    ref_model_id=None,
    cat_group_cols=None,
    cont_group_cols=None,
    cont_group_bins=None,
    include_coarse_ancestry=False,
    pc_clusters=None,
    metrics=None,
    evaluate_base_models=True,
    min_group_size=30,
):
    if cont_group_cols is not None:
        assert cont_group_bins is not None, (
            "Bins must be provided for continuous group columns!"
        )

    prs_dataset.set_backend("numpy")

    if trained_models is None or len(trained_models) == 0:
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
        ref_model_id=ref_model_id,
        metrics=metrics,
        evaluate_base_models=evaluate_base_models,
        eval_category="All",
        eval_group="All",
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
                    ref_model_id=ref_model_id,
                    mask=msk,
                    min_group_size=min_group_size,
                    metrics=metrics,
                    evaluate_base_models=evaluate_base_models,
                    eval_category=mg,
                    eval_group=m,
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
    ref_model_id=None,
    mask=None,
    metrics=None,
    evaluate_base_models=True,
    min_group_size=30,
    eval_category="All",
    eval_group="All",
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
        if num_cases < 10 or num_cases > phenotype.shape[0] - 10:
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

        for metric in metric_names:
            try:
                if metric in INCREMENTAL_METRICS:
                    value = EVAL_METRICS[metric](
                        phenotype_values, pred_values, covar.loc[keep, :]
                    )
                else:
                    value = EVAL_METRICS[metric](phenotype_values, pred_values)

                if "R2" in metric:
                    try:
                        se = r2_stats(value, n)["SE"]
                    except AssertionError:
                        se = np.nan
                else:
                    se = np.nan

            except Exception as e:
                print(f"Error evaluating metric {metric} on model {model_id}: {e}")
                value = np.nan
                se = np.nan

            records.append(
                {
                    **row_meta,
                    "metric": metric,
                    "metric_kind": "base",
                    "ref_model_id": None,
                    "ref_model_name": None,
                    "value": value,
                    "se": se,
                    "n": n,
                    "eval_category": eval_category,
                    "eval_group": eval_group,
                }
            )

            # Additional incremental layer vs reference model:
            if (
                metric in INCREMENTAL_METRICS
                and ref_model_id is not None
                and ref_model_id in model_ids
                and model_id != ref_model_id
                and row_meta.get("prediction_type", "full") == "full"
            ):
                ref_keep = ~np.isnan(prs_df[ref_model_id].values)
                keep_both = keep & ref_keep
                n_both = int(keep_both.sum())
                if n_both >= min_group_size:
                    try:
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
                    delta_se = np.nan
                    if "R2" in metric:
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
                            "value": delta_val,
                            "se": delta_se,
                            "n": n_both,
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
                "train_source",
                "metric",
                "metric_kind",
                "ref_model_id",
                "ref_model_name",
                "value",
                "se",
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
        help="The path to the test data file.",
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
        default=30,
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
        "--include-coarse-ancestry",
        dest="include_coarse_ancestry",
        action="store_true",
        default=False,
        help="Whether to include coarse ancestry groupings in the evaluation metrics.",
    )
    parser.add_argument(
        "--trained-models-suffix",
        dest="trained_models_suffix",
        type=str,
        default="",
        help="Optional suffix used during training (loads from trained_models_<suffix>/...).",
    )

    args = parser.parse_args()

    print(f"Evaluating Meta PRS performance on {args.test_data}")

    prs_dataset = PRSDataset.from_pickle(args.test_data)

    # Obtain path for relevant trained models:

    trained_root = "trained_models"
    if args.trained_models_suffix and len(args.trained_models_suffix.strip()) > 0:
        trained_root = f"trained_models_{args.trained_models_suffix.strip()}"

    trained_models_root = osp.dirname(
        osp.dirname(args.test_data.replace("harmonized_data", trained_root))
    )

    if args.model_source is None:
        # existing behavior (search all subfolders)
        trained_models_path = osp.join(trained_models_root, "*", "*", "*.pkl")
    else:
        # filter subfolder by pattern
        trained_models_path = osp.join(
            trained_models_root, "*", f"{args.model_source}", "*.pkl"
        )

    print(f"> Loading trained models from: {trained_models_path}")

    trained_models = {}
    model_catalog = []
    test_biobank = osp.basename(osp.dirname(args.test_data))

    for f in glob.glob(trained_models_path):
        model_basename = osp.basename(f).replace(".pkl", "")
        split_fname = f.split("/")
        model_prefix = split_fname[-3] + "/" + split_fname[-2] + ":"
        model_name = model_prefix + model_basename
        model_catalog.append(_parse_model_id(model_name))

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
    ref_model_id = _resolve_reference_model_id(
        model_catalog, test_biobank=test_biobank, ref_model_name="Covariates"
    )

    eval_df = stratified_evaluation(
        prs_dataset,
        trained_models,
        model_catalog=model_catalog,
        ref_model_id=ref_model_id,
        cat_group_cols=args.cat_group_cols,
        cont_group_cols=args.cont_group_cols,
        cont_group_bins=args.cont_group_bins,
        include_coarse_ancestry=args.include_coarse_ancestry,
        pc_clusters=args.pc_clusters,
        min_group_size=args.min_group_size,
    )

    # Canonical test metadata for downstream plotting/filtering.
    eval_df["analysis_id"] = args.test_data.split("/")[-3]
    eval_df["test_biobank"] = test_biobank
    eval_df["test_dataset"] = osp.basename(args.test_data).replace(".pkl", "")

    eval_root = "evaluation"
    if args.trained_models_suffix and len(args.trained_models_suffix.strip()) > 0:
        eval_root = f"evaluation_{args.trained_models_suffix.strip()}"

    output_path = args.test_data.replace("harmonized_data", eval_root).replace(
        ".pkl", ".csv"
    )

    print("> Saving evaluation metrics to:\n\t", output_path)

    makedir(os.path.dirname(output_path))
    eval_df.to_csv(output_path, index=False)
