import glob
import os
import os.path as osp
import sys

import pandas as pd
import seaborn as sns

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))
sys.path.append(osp.join(parent_dir, "evaluation/"))

from model_utils import (
    get_analysis_id_mapper,
    get_analysis_to_table_mapper,
    get_model_name_mapper,
)

# --------------------------------------------------------


MODEL_NAME_MAP = get_model_name_mapper()
ANALYSIS_TO_PHENOTYPE_MAP = get_analysis_id_mapper(target_col="Phenotype")
ANALYSIS_TO_SHORT_PHENOTYPE_MAP = get_analysis_id_mapper(target_col="Phenotype_short")
ANALYSIS_TO_TABLE_MAP = get_analysis_to_table_mapper()

SEX_LABEL_MAP = {"0": "Female", "1": "Male"}

# --------------------------------------------------------

BIOBANK_NAME_MAP = {
    "ukbb": "UK Biobank",
    "cartagene": "CARTaGENE Biobank",
}

BIOBANK_NAME_MAP_SHORT = {
    "ukbb": "UKB",
    "cartagene": "CaG",
}

# --------------------------------------------------------

PRS_NAME_COLOR_MAP = {
    "T2D": "#FDB462",
    "BMI": "#B3DE69",
    "HbA1c": "#FFED6F",
    "FG": "#66C2A5",
    "TG": "#BC80BD",
    "T1D": "#FCCDE5",
    "HYPO": "#CAB2D6",
    "CAD": "#80B1D3",
    "LDL": "#FDBF6F",
    "SBP": "#FCCDE5",
    "SMK": "#D9D9D9",
    "HTN": "#FDB462",
    "DBP": "#C6DBEF",
    "HF": "#FB8072",
    "AF": "#BC80BD",
    "HEIGHT": "#CCEBC5",
    "STR_433": "#8DD3C7",
    "STR_433.1": "#FFED6F",
    "HDL": "#B39DDB",
    "DEM": "#C6DBEF",
    "EDU": "#9ADBE8",
    "GOUT": "#E5C494",
    "URATE": "#FFE082",
    "ASTHMA": "#66C2A5",
    "EOS": "#FFFFB3",
    "ALLERGY": "#E78AC3",
}

# --------------------------------------------------------
# Sorting lists:

SORTED_ANCESTRY_LABEL = ["All", "EUR", "MID", "CSA", "EAS", "AMR", "AFR", "OTH"]
SORTED_COARSE_ANCESTRY_LABEL = ["All", "EUR", "non-EUR"]

UKBB_SORTED_UMAP_CLUSTERS = [
    "All",
    "17 ENG-BRI",
    "20 ENG-BRI",
    "21 ENG-BRI-OTH",
    "24 ENG-BRI-OTH",
    "16 ESPPOR",
    "1 ITA",
    "11 ENG-MIX",
    "3 ENG-BRI",
    "5 LEV",
    "9 SAS-MIX",
    "6 NAF",
    "2 FIN",
    "4 ENG-BRI-AOW",
    "25 ENG-AFR-CAR-MIX",
    "7 SAS",
    "23 HAFR",
    "14 ENG-EAS-MIX",
    "22 ENG-CAR-WAB",
    "8 SAS-IND",
    "10 SOM",
    "12 AMR",
    "15 NEP",
    "13 SEA-CHN-OTH",
    "0 JPN",
    "19 WAFR-CAR",
    "18 AFR",
]

CARTAGENE_SORTED_UMAP_CLUSTERS = [
    "All",
    "14-FRC",
    "13-FRC",
    "10-CAN-FRC",
    "12-CAN-FRC",
    "11-MED",
    "7-EER",
    "9-EUR-JEW",
    "5-MIE",
    "4-NAF",
    "8-EER-JEW",
    "2-SAS",
    "6-AFR-EUR",
    "3-CSA",
    "1-EAS",
    "0-HAI-CAR",
]

# --------------------------------------------------------

METRIC_NAME_MAP = {
    "Incremental_R2": "Incremental $R^2$",
    "Liability_R2": "Liability $R^2$",
    "Nagelkerke_R2": "Nagelkerke $R^2$",
    "CoxSnell_R2": "Cox-Snell $R^2",
    "McFadden_R2": "McFadden $R^2",
    "Liability_Probit_R2": "Liability $R^2",
    "Liability_Logit_R2": "Liability $R^2",
    "AUROC": "ROC AUC",
    "AUPRC": "PR AUC",
    "Pearson_R": "Pearson $R$",
    "Partial_Correlation": "Partial Pearson $R$",
}

# --------------------------------------------------------
# Helper functions:


def assign_ancestry_consistent_colors(groups, palette="Set3"):
    """
    Assign consistent colors to the ancestry groups for plotting.
    :param groups: A list of ancestry group names
    :param palette: The color palette to use
    :return: A dictionary of group names and colors
    """
    import seaborn as sns

    if isinstance(groups, str):
        groups = [groups]

    colors = sns.color_palette(palette, len(SORTED_ANCESTRY_LABEL[1:]))
    color_dict = dict(zip(SORTED_ANCESTRY_LABEL[1:], colors))

    return {k: color_dict[k] for k in groups if k in color_dict}


def assign_models_consistent_colors(models, palette="Set3"):
    """
    Assign consistent colors to the models for plotting.
    :param models: A list of model names
    :param palette: The color palette to use
    :return: A dictionary of model names and colors
    """

    if isinstance(models, str):
        models = [models]

    baseline_ancestry_model_names = [
        "EUR",
        "EAS",
        "CSA",
        "AFR",
        "ALL",
        "AMR",
        "FIXEDPRS",
    ]
    ancestry_colors = sns.color_palette(palette, len(baseline_ancestry_model_names))

    colors = dict(zip(baseline_ancestry_model_names, ancestry_colors))

    colors["Male"] = "#A1BE95"
    colors["Female"] = "#F98866"
    colors["MoEPRS"] = "#375E97"
    colors["MultiPRS"] = "#FFBB00"
    colors.update(PRS_NAME_COLOR_MAP)

    all_unique_models = set(
        [v for inner in MODEL_NAME_MAP.values() for v in inner.values()]
    )

    remaining_models = sorted(list(all_unique_models - set(colors.keys())))
    remaining_colors = sns.color_palette("husl", len(remaining_models))

    colors.update(dict(zip(remaining_models, remaining_colors)))

    unknown_models = sorted(list(set(models) - set(colors.keys())))
    if unknown_models:
        unknown_colors = sns.color_palette("pastel", len(unknown_models))
        colors.update(dict(zip(unknown_models, unknown_colors)))

    return {m: colors[m] for m in models}


def sort_groups(groups):
    if "non-EUR" in groups:
        return sorted(groups, key=lambda x: SORTED_COARSE_ANCESTRY_LABEL.index(x))
    if len(set(groups).intersection(SORTED_ANCESTRY_LABEL)) > 2:
        return sorted(groups, key=lambda x: SORTED_ANCESTRY_LABEL.index(x))
    elif len(set(groups).intersection(UKBB_SORTED_UMAP_CLUSTERS)) > 2:
        return sorted(groups, key=lambda x: UKBB_SORTED_UMAP_CLUSTERS.index(x))
    elif len(set(groups).intersection(CARTAGENE_SORTED_UMAP_CLUSTERS)) > 2:
        return sorted(groups, key=lambda x: CARTAGENE_SORTED_UMAP_CLUSTERS.index(x))
    else:
        return sorted(groups)


def _filter_by_ref_model_biobank(metrics_df, ref_model_biobank):
    if "ref_model_biobank" not in metrics_df.columns:
        raise ValueError(
            "Reference-biobank filtering requested, but 'ref_model_biobank' "
            "is not in metrics_df."
        )

    ref_model_biobank = (
        [ref_model_biobank]
        if isinstance(ref_model_biobank, str)
        else ref_model_biobank
    )
    ref_model_biobank = list(ref_model_biobank)

    keep = pd.Series(False, index=metrics_df.index)
    literal = [
        x for x in ref_model_biobank if x not in ("train_biobank", "test_biobank")
    ]
    if literal:
        keep |= metrics_df["ref_model_biobank"].isin(literal)
    if "train_biobank" in ref_model_biobank:
        keep |= metrics_df["ref_model_biobank"].eq(metrics_df["train_biobank"])
    if "test_biobank" in ref_model_biobank:
        keep |= metrics_df["ref_model_biobank"].eq(metrics_df["test_biobank"])

    return metrics_df.loc[keep].copy()


def _cross_validation_metric_files(file_path):
    """Return fold-level metric files corresponding to ``file_path``."""
    file_path = osp.normpath(file_path)
    parent_dir = osp.dirname(file_path)

    if osp.basename(parent_dir).startswith("fold_"):
        biobank_dir = osp.dirname(parent_dir)
    else:
        biobank_dir = parent_dir

    return sorted(
        glob.glob(osp.join(biobank_dir, "fold_*", osp.basename(file_path)))
    )


def evaluation_metrics_exist(file_path):
    """Return whether legacy or fold-level evaluation metrics exist."""
    return osp.exists(file_path) or bool(_cross_validation_metric_files(file_path))


def generate_external_cross_validation_metrics(
    file_path,
    train_biobank="ukbb",
    metrics=None,
    n_bootstrap=1000,
    bootstrap_ci=0.95,
    random_state=42,
):
    """
    Generate bootstrapped metrics for a fold-ensemble on external full data.

    Individual-level predictions are averaged across all training-fold models
    before metrics are computed. External participants are then resampled to
    estimate sampling uncertainty. The ensemble table is persisted at the
    requested root-level ``full_data.csv`` path; legacy per-fold files are left
    untouched but are superseded by this table when plotting.
    """
    file_path = osp.normpath(file_path)
    if osp.basename(file_path) != "full_data.csv":
        raise ValueError(
            "On-the-fly external evaluation only supports full_data.csv targets."
        )

    test_biobank_dir = osp.dirname(file_path)
    test_biobank = osp.basename(test_biobank_dir)
    analysis_id = osp.basename(osp.dirname(test_biobank_dir))

    if isinstance(metrics, str):
        metrics = [metrics]
    elif metrics is not None:
        metrics = list(metrics)

    output_file = (
        file_path if osp.isabs(file_path) else osp.join(parent_dir, file_path)
    )
    if osp.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
        except (OSError, ValueError):
            existing_df = pd.DataFrame()
        is_bootstrap_ensemble = (
            not existing_df.empty
            and "evaluation_scope" in existing_df
            and existing_df["evaluation_scope"]
            .eq("external_ensemble_bootstrap")
            .all()
            and "uncertainty_method" in existing_df
            and existing_df["uncertainty_method"]
            .eq("participant_bootstrap")
            .all()
        )
        available_metrics = (
            set(existing_df["metric"].dropna())
            if "metric" in existing_df
            else set()
        )
        if is_bootstrap_ensemble and (
            metrics is None or set(metrics).issubset(available_metrics)
        ):
            return [file_path]

    model_fold_dirs = sorted(
        glob.glob(
            osp.join(
                parent_dir,
                "data",
                "trained_models",
                analysis_id,
                train_biobank,
                "fold_*",
            )
        )
    )
    if not model_fold_dirs:
        raise FileNotFoundError(
            "Cannot generate external metrics: no fold models found beneath "
            f"data/trained_models/{analysis_id}/{train_biobank}/."
        )

    dataset_path = osp.join(
        parent_dir,
        "data",
        "harmonized_data",
        analysis_id,
        test_biobank,
        "full_data.pkl",
    )
    if not osp.exists(dataset_path):
        raise FileNotFoundError(
            f"Cannot generate external metrics: dataset not found at {dataset_path}"
        )

    model_folds = [osp.basename(path) for path in model_fold_dirs]
    print(
        f"> Generating bootstrapped external ensemble metrics for {analysis_id} "
        f"({test_biobank}; averaging predictions from folds: "
        f"{', '.join(model_folds)}; metrics: "
        f"{', '.join(metrics) if metrics is not None else 'all'})"
    )

    # Lazy imports keep the lightweight plotting helpers inexpensive for plots
    # that only read existing evaluation CSVs.
    from baseline_models import (
        AncestryWeightedPRS,
        AttributePartitionedPRS,
        MultiPRS,
    )
    from evaluate_predictive_performance import (
        average_fold_predictions,
        stratified_evaluation,
    )
    from moe import MoEPRS
    from moe_pytorch import TorchMoEPRS
    from PRSDataset import PRSDataset

    prs_dataset = PRSDataset.from_pickle(dataset_path)
    trained_models = {}
    trained_analysis_root = osp.join(
        parent_dir,
        "data",
        "trained_models",
        analysis_id,
    )
    model_files = sorted(
        glob.glob(
            osp.join(
                trained_analysis_root,
                train_biobank,
                "fold_*",
                "*",
                "*.pkl",
            )
        )
    )
    for model_file in model_files:
        path_parts = osp.relpath(model_file, trained_analysis_root).split(os.sep)
        if len(path_parts) != 4:
            raise ValueError(f"Unexpected trained model path: {model_file}")

        model_biobank, model_fold, model_source, model_filename = path_parts
        model_basename = osp.splitext(model_filename)[0]
        model_id = (
            f"{model_biobank}/{model_fold}/{model_source}:{model_basename}"
        )

        if "moe" in model_id.lower():
            if "torch" in model_id.lower():
                trained_models[model_id] = TorchMoEPRS.from_saved_model(model_file)
            else:
                trained_models[model_id] = MoEPRS.from_saved_model(model_file)
        elif "AncestryWeightedPRS" in model_id:
            trained_models[model_id] = AncestryWeightedPRS.from_saved_model(
                model_file
            )
        elif "SexMatchedPRS" in model_id:
            trained_models[model_id] = AttributePartitionedPRS.from_saved_model(
                model_file
            )
        else:
            trained_models[model_id] = MultiPRS.from_saved_model(model_file)

    if not trained_models:
        raise FileNotFoundError(
            f"No trained models found for {analysis_id}/{train_biobank}."
        )

    ensemble_predictions, ensemble_catalog = average_fold_predictions(
        prs_dataset, trained_models
    )
    eval_df = stratified_evaluation(
        prs_dataset,
        trained_predictions=ensemble_predictions,
        model_catalog=ensemble_catalog,
        test_biobank=test_biobank,
        cat_group_cols=["Sex"] if "_SEX" in analysis_id else None,
        coarse_ancestry_only=True,
        metrics=metrics,
        evaluate_base_models=False,
        bootstrap=True,
        n_bootstrap=n_bootstrap,
        bootstrap_ci=bootstrap_ci,
        random_state=random_state,
    )
    eval_df["analysis_id"] = analysis_id
    eval_df["test_biobank"] = test_biobank
    eval_df["test_dataset"] = "full_data"
    eval_df["evaluation_scope"] = "external_ensemble_bootstrap"
    eval_df["test_fold"] = "ensemble"

    if metrics is not None and osp.exists(output_file):
        existing_df = pd.read_csv(output_file)
        existing_is_ensemble = (
            not existing_df.empty
            and "evaluation_scope" in existing_df
            and existing_df["evaluation_scope"]
            .eq("external_ensemble_bootstrap")
            .all()
        )
        if existing_is_ensemble:
            existing_df = existing_df.loc[~existing_df["metric"].isin(metrics)]
            eval_df = pd.concat([existing_df, eval_df], ignore_index=True)

    os.makedirs(osp.dirname(output_file), exist_ok=True)
    eval_df.to_csv(output_file, index=False)
    return [file_path]


def aggregate_cross_validation_metrics(eval_df):
    """
    Aggregate held-out metrics across cross-validation folds.

    ``value`` is the arithmetic mean across folds and ``se`` is the standard
    error of that mean (fold-level sample SD divided by sqrt(K)). Analytical
    standard errors stored in the individual evaluation files are deliberately
    ignored. ``n`` is summed for disjoint held-out folds and retained as the
    per-evaluation cohort size for repeated full-cohort external validation.
    """
    if (
        "evaluation_scope" in eval_df
        and not eval_df.empty
        and eval_df["evaluation_scope"]
        .eq("external_ensemble_bootstrap")
        .all()
    ):
        # These rows already describe one fold-prediction ensemble, and their
        # SE comes from participant resampling. Treating ``ensemble`` as a fold
        # would overwrite the bootstrap uncertainty with a one-value fold SE.
        return eval_df.copy()

    fold_cols = [col for col in ("test_fold", "train_fold") if col in eval_df]
    if len(fold_cols) == 0:
        return eval_df.copy()

    excluded_cols = {"value", "se", "n", *fold_cols}
    group_cols = [col for col in eval_df.columns if col not in excluded_cols]

    def fold_standard_error(values):
        values = values.dropna()
        if len(values) < 2:
            return float("nan")
        return values.std(ddof=1) / (len(values) ** 0.5)

    summary = (
        eval_df.groupby(group_cols, dropna=False, sort=False)
        .agg(
            value=("value", "mean"),
            se=("value", fold_standard_error),
            n_sum=("n", "sum"),
            n_max=("n", "max"),
            n_folds=("value", "count"),
        )
        .reset_index()
    )

    if "evaluation_scope" in summary.columns:
        is_external = summary["evaluation_scope"].eq("external_full")
    else:
        is_external = pd.Series(False, index=summary.index)

    summary["n"] = summary["n_sum"]
    summary.loc[is_external, "n"] = summary.loc[is_external, "n_max"]

    return summary.drop(columns=["n_sum", "n_max"])


def read_transform_eval_metrics(
    file_path,
    generate_missing_external=False,
    train_biobank="ukbb",
    external_metrics=None,
):
    if generate_missing_external:
        generate_external_cross_validation_metrics(
            file_path,
            train_biobank=train_biobank,
            metrics=external_metrics,
        )

    fold_files = _cross_validation_metric_files(file_path)
    root_eval_df = pd.read_csv(file_path) if osp.exists(file_path) else None
    use_external_ensemble = (
        root_eval_df is not None
        and not root_eval_df.empty
        and "evaluation_scope" in root_eval_df
        and root_eval_df["evaluation_scope"]
        .eq("external_ensemble_bootstrap")
        .all()
    )

    if use_external_ensemble:
        eval_df = root_eval_df
    elif fold_files:
        fold_dfs = []
        for fold_file in fold_files:
            fold_df = pd.read_csv(fold_file)
            if "test_fold" not in fold_df.columns:
                fold_df["test_fold"] = osp.basename(osp.dirname(fold_file))
            fold_dfs.append(fold_df)

        eval_df = aggregate_cross_validation_metrics(
            pd.concat(fold_dfs, ignore_index=True)
        )
    else:
        # Backwards compatibility for evaluation results from a single split.
        if root_eval_df is None:
            raise FileNotFoundError(file_path)
        eval_df = root_eval_df

    analysis_id = eval_df["analysis_id"].iloc[0] if len(eval_df) > 0 else None
    name_map = MODEL_NAME_MAP.get(analysis_id, {})

    required = [
        "analysis_id",
        "test_biobank",
        "test_dataset",
        "model_id",
        "model_name",
        "prediction_type",
        "model_category",
        "train_biobank",
        "train_source",
        "metric",
        "metric_kind",
        "value",
        "se",
        "n",
        "eval_category",
        "eval_group",
    ]
    missing = [c for c in required if c not in eval_df.columns]
    if missing:
        raise ValueError(
            f"Expected long-format evaluation file. Missing columns: {missing}"
        )

    eval_df["phenotype"] = eval_df["analysis_id"].map(
        lambda x: ANALYSIS_TO_PHENOTYPE_MAP.get(x, x)
    )
    eval_df["model_name"] = eval_df["model_name"].map(
        lambda x: name_map.get(x.replace("-covariates", ""), x)
    )

    # ----------------------------------------------------------------------
    # Clean up names of evaluation cohorts:

    def map_sex_label(x):
        try:
            return SEX_LABEL_MAP[x]
        except Exception:
            return x

    eval_df["eval_group"] = eval_df["eval_group"].astype(str).apply(map_sex_label)

    return eval_df


def postprocess_metrics_df(
    metrics_df,
    metric,
    metric_kind="base",
    ref_model_biobank=None,
    prediction_type=None,
    category="Coarse Ancestry",
    min_sample_size=100,
    aggregate_single_prs=True,
    add_training_biobank_to_model_name=False,
):

    # Sanity checks:
    assert metric_kind in ("incremental_vs_ref", "base")
    if ref_model_biobank is not None:
        assert metric_kind == "incremental_vs_ref", (
            "ref_model_biobank filtering only applies when "
            "metric_kind='incremental_vs_ref'."
        )

    # ------------------------------------------------------
    # Filter by metric from long-format table
    if prediction_type is None:
        prediction_type = ["prs_only", "full"][
            metric_kind == "incremental_vs_ref"
        ]
    if prediction_type not in {"prs_only", "full"}:
        raise ValueError("prediction_type must be either 'prs_only' or 'full'.")

    sub_metrics_df = metrics_df.loc[
        (metrics_df["metric"] == metric)
        & (metrics_df["metric_kind"] == metric_kind)
        & (
            metrics_df["model_category"].isin(["SinglePRS", "Covariates"])
            | (metrics_df["prediction_type"] == prediction_type)
        )
    ].copy()

    if ref_model_biobank is not None:
        sub_metrics_df = _filter_by_ref_model_biobank(
            sub_metrics_df, ref_model_biobank
        )

    # ------------------------------------------------------
    # Transform to wide-table format:

    sub_metrics_df[metric] = sub_metrics_df["value"]
    sub_metrics_df[f"{metric}_err"] = sub_metrics_df["se"]

    sub_metrics_df["Model Name"] = sub_metrics_df["model_name"].astype("string")

    if add_training_biobank_to_model_name:
        bb_suffix = (
            sub_metrics_df["train_biobank"]
            .map(lambda x: BIOBANK_NAME_MAP_SHORT.get(x, x) if pd.notna(x) else x)
            .astype("string")
        )
        has_source = bb_suffix.notna() & (bb_suffix.str.len() > 0)
        sub_metrics_df.loc[has_source, "Model Name"] += (
            " (" + bb_suffix[has_source] + ")"
        )

    sub_metrics_df["Evaluation Group"] = sub_metrics_df["eval_group"]
    sub_metrics_df["PGS"] = sub_metrics_df["model_id"]

    # ------------------------------------------------------
    # Filter the metrics dataframe:
    sub_metrics_df = sub_metrics_df.loc[
        sub_metrics_df.eval_category.isin([category, "All"])
    ]
    # Remove entries with tiny sample sizes:
    sub_metrics_df = sub_metrics_df.loc[sub_metrics_df.n >= min_sample_size]
    # Remove entries with NaNs:
    sub_metrics_df = sub_metrics_df.loc[~sub_metrics_df["value"].isna()]

    # ------------------------------------------------------
    # Filter based on model categories:
    if "SinglePRS+Covariates" in sub_metrics_df["model_category"].unique():
        single_model_label = "SinglePRS+Covariates"
    else:
        single_model_label = "SinglePRS"

    model_cats = [
        "MoE",
        "MultiPRS",
        "AncestryWeightedPRS",
        "AttributePartitionedPRS",
        single_model_label,
    ]

    if metric in (
        "AUPRC",
        "AUROC",
        "PR_AUC",
        "ROC_AUC",
        "MSE",
        "Pearson_R",
    ):
        model_cats.append("Covariates")

    sub_metrics_df = sub_metrics_df.loc[
        sub_metrics_df["model_category"].isin(model_cats)
    ]

    # ------------------------------------------------------
    # Aggregate single-source PRS:
    if aggregate_single_prs:
        # Get entries for SinglePRS methods:
        mask = (sub_metrics_df["model_category"] == single_model_label) & (
            sub_metrics_df["Model Name"] != f"{category}-matched PRS"
        )

        if mask.sum() > 1:
            grouped = sub_metrics_df.loc[mask].groupby("eval_group")
            if metric.endswith("MSE"):
                single_prs_agg = grouped.apply(lambda x: x.loc[x["value"].idxmin()])
            else:
                single_prs_agg = grouped.apply(lambda x: x.loc[x["value"].idxmax()])

            single_prs_agg = single_prs_agg.reset_index(drop=True)
            single_prs_agg["model_name"] = "Best Single Source PRS"
            single_prs_agg["Model Name"] = "Best Single Source PRS"

            sub_metrics_df = pd.concat(
                [
                    single_prs_agg.reset_index(drop=True),
                    sub_metrics_df.loc[~mask].reset_index(drop=True),
                ],
                ignore_index=True,
            )

    return sub_metrics_df


def extract_accuracy_data_all_phenotypes(
    moe_model_name,
    test_biobank,
    train_biobank=None,
    ref_model_biobank=None,
    dataset="test_data",
    analysis_table_id="multi_ancestry_prs_table",
    binary_metric="Nagelkerke_R2",
    metric_kind="base",
    prediction_type=None,
    keep_analyses=None,
    exclude_analyses=None,
    exclude_all_group=True,
    add_training_biobank_to_model_name=False,
    generate_missing_external=False,
    external_metric=None,
    aggregate_single_prs=True,
):
    analysis_results = []

    for d in glob.glob(f"data/harmonized_data/*/{test_biobank}"):
        analysis_id = d.split("/")[-2]
        if ANALYSIS_TO_TABLE_MAP.get(analysis_id) != analysis_table_id:
            continue

        if keep_analyses is not None:
            if analysis_id not in keep_analyses:
                continue

        if exclude_analyses is not None:
            if analysis_id in exclude_analyses:
                continue

        eval_file = f"data/evaluation/{analysis_id}/{test_biobank}/{dataset}.csv"
        if not generate_missing_external and not evaluation_metrics_exist(eval_file):
            print(
                f"> Skipping {analysis_id}/{test_biobank}: no evaluation "
                f"metrics found for {dataset} (root or fold-level).",
                file=sys.stderr,
            )
            continue

        analysis_results.append(
            extract_accuracy_data(
                moe_model_name,
                analysis_id,
                test_biobank,
                train_biobank=train_biobank,
                ref_model_biobank=ref_model_biobank,
                binary_metric=binary_metric,
                metric_kind=metric_kind,
                prediction_type=prediction_type,
                dataset=dataset,
                exclude_all_group=exclude_all_group,
                add_training_biobank_to_model_name=add_training_biobank_to_model_name,
                generate_missing_external=generate_missing_external,
                external_metric=external_metric,
                aggregate_single_prs=aggregate_single_prs,
            )
        )

    if len(analysis_results) == 0:
        raise FileNotFoundError(
            "No evaluation metrics were found for the requested analyses: "
            f"table={analysis_table_id}, biobank={test_biobank}, dataset={dataset}."
        )

    df = pd.concat(analysis_results).reset_index(drop=True)

    # Simplify the ancestry weighted model:
    if test_biobank == "ukbb":
        df = df.loc[df["Model Name"] != "Ancestry-weighted PRS (CaG)"]
        df["Model Name"] = df["Model Name"].replace(
            "Ancestry-weighted PRS (UKB)", "Ancestry-weighted PRS"
        )
    else:
        df = df.loc[df["Model Name"] != "Ancestry-weighted PRS (UKB)"]
        df["Model Name"] = df["Model Name"].replace(
            "Ancestry-weighted PRS (CaG)", "Ancestry-weighted PRS"
        )

    return df


def extract_accuracy_data(
    moe_model_name,
    analysis_id,
    test_biobank,
    train_biobank=None,
    ref_model_biobank=None,
    metric="Incremental_R2",
    binary_metric="Nagelkerke_R2",
    metric_kind="base",
    prediction_type=None,
    dataset="test_data",
    evaluation_category="Coarse Ancestry",
    exclude_all_group=True,
    add_training_biobank_to_model_name=False,
    generate_missing_external=False,
    external_metric=None,
    aggregate_single_prs=True,
):
    # Extract accuracy metrics:
    f = f"data/evaluation/{analysis_id}/{test_biobank}/{dataset}.csv"
    df = read_transform_eval_metrics(
        f,
        generate_missing_external=generate_missing_external,
        train_biobank=train_biobank or "ukbb",
        external_metrics=external_metric,
    )

    df = df.loc[
        (df["model_category"] != "MoE")
        | df["model_name"].isin(
            [
                f"{moe_model_name}",
            ]
        )
    ]

    # For non-trained models, restrict to same biobank:
    df = df.loc[
        (df["model_category"] != "SinglePRS") | (df["train_biobank"] == test_biobank)
    ]

    if train_biobank is not None:
        df = df.loc[df["train_biobank"] == train_biobank]

    # Rename the ensemble models for clarity:
    df["model_name"] = df["model_name"].replace(
        {
            moe_model_name: "MoEPRS",
            "MultiPRS": "MultiPRS",
            "AncestryWeightedPRS": "Ancestry-weighted PRS",
        }
    )

    # Correction for binary phenotypes:
    if metric == "Incremental_R2" and binary_metric in set(df["metric"].unique()):
        post_metric = binary_metric
    else:
        post_metric = metric

    dfs = postprocess_metrics_df(
        df,
        post_metric,
        metric_kind=metric_kind,
        ref_model_biobank=ref_model_biobank,
        prediction_type=prediction_type,
        category=evaluation_category,
        min_sample_size=50,
        aggregate_single_prs=aggregate_single_prs,
        add_training_biobank_to_model_name=add_training_biobank_to_model_name,
    )

    if post_metric == binary_metric:
        dfs[metric] = dfs[post_metric]
        if f"{post_metric}_err" in dfs.columns:
            dfs[f"{metric}_err"] = dfs[f"{post_metric}_err"]

    if exclude_all_group:
        dfs = dfs.loc[dfs["Evaluation Group"] != "All"]

    dfs["Phenotype"] = ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]

    return dfs


def reshape_eval_long_to_plot_wide(
    eval_df,
    analysis_id=None,
    metric_kind="base",
    ref_model_biobank=None,
    model_col="model_name",
    index_cols=None,
):
    """
    Convert long-format evaluation output to a wide plotting dataframe.

    Output includes:
      - PGS
      - EvalGroup
      - EvalCategory (if present)
      - one column per metric with values from `value`
      - optional <metric>_err columns when `se` is available
    """

    required = {"metric", "value", "eval_group"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(
            f"Expected long-format evaluation table. Missing columns: {sorted(missing)}"
        )

    out = eval_df.copy()

    if metric_kind is not None:
        if "metric_kind" not in out.columns:
            raise ValueError(
                "metric_kind filtering requested but 'metric_kind' is not in eval_df."
            )
        out = out.loc[out["metric_kind"] == metric_kind].copy()

    if ref_model_biobank is not None:
        if metric_kind != "incremental_vs_ref":
            raise ValueError(
                "ref_model_biobank filtering only applies when "
                "metric_kind='incremental_vs_ref'."
            )

        out = _filter_by_ref_model_biobank(out, ref_model_biobank)

    name_map = MODEL_NAME_MAP.get(analysis_id, {}) if analysis_id is not None else {}
    out["PGS"] = out[model_col].map(lambda x: name_map.get(x, x))

    if index_cols is None:
        index_cols = ["PGS", "eval_group"]
        if "eval_category" in out.columns:
            index_cols.append("eval_category")
        for c in ("model_id", "n"):
            if c in out.columns:
                index_cols.append(c)

    val_wide = out.pivot_table(
        index=index_cols,
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()

    merged = val_wide
    if "se" in out.columns:
        err_wide = out.pivot_table(
            index=index_cols,
            columns="metric",
            values="se",
            aggfunc="first",
        ).reset_index()
        err_wide.columns = [
            c if c in index_cols else f"{c}_err" for c in err_wide.columns
        ]
        merged = val_wide.merge(err_wide, on=index_cols, how="left")

    return merged.rename(
        columns={"eval_group": "EvalGroup", "eval_category": "EvalCategory"}
    )
