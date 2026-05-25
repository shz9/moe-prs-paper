import os.path as osp
import sys

import pandas as pd
import seaborn as sns

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))

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
    "ROC_AUC": "ROC AUC",
    "PR_AUC": "PR AUC",
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


def read_transform_eval_metrics(file_path):

    eval_df = pd.read_csv(file_path)

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
    eval_df["model_name"] = eval_df["model_name"].map(lambda x: name_map.get(x, x))

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
    category="Ancestry",
    min_sample_size=100,
    aggregate_single_prs=True,
    include_cohort_matched=False,
):

    # Sanity checks:
    assert metric_kind in ("incremental_vs_ref", "base")

    # ------------------------------------------------------
    # Filter by metric from long-format table
    sub_metrics_df = metrics_df.loc[
        (metrics_df["metric"] == metric)
        & (metrics_df["metric_kind"] == metric_kind)
        & (
            metrics_df["model_category"].isin(["SinglePRS", "Covariates"])
            | (
                metrics_df["prediction_type"]
                == ["prs_only", "full"][metric_kind == "incremental_vs_ref"]
            )
        )
    ].copy()

    # ------------------------------------------------------
    # Transform to wide-table format:

    sub_metrics_df[metric] = sub_metrics_df["value"]
    sub_metrics_df[f"{metric}_err"] = sub_metrics_df["se"]

    bb_suffix = (
        sub_metrics_df["train_biobank"]
        .map(lambda x: BIOBANK_NAME_MAP_SHORT.get(x, x) if pd.notna(x) else x)
        .astype("string")
    )
    model_name_with_source = sub_metrics_df["model_name"].astype("string")
    has_source = bb_suffix.notna() & (bb_suffix.str.len() > 0)
    sub_metrics_df["Model Name"] = model_name_with_source
    sub_metrics_df.loc[has_source, "Model Name"] = (
        model_name_with_source[has_source] + " (" + bb_suffix[has_source] + ")"
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

    if metric in ("PR_AUC", "ROC_AUC", "MSE", "Pearson_R"):
        model_cats.append("Covariates")

    sub_metrics_df = sub_metrics_df.loc[
        sub_metrics_df["model_category"].isin(model_cats)
    ]

    # ------------------------------------------------------
    # Include cohort-matched PRS

    if include_cohort_matched:
        mask = (sub_metrics_df["model_category"] == single_model_label) & (
            sub_metrics_df["eval_group"] == sub_metrics_df["model_name"]
        )

        if mask.sum() > 1:

            matched_df = sub_metrics_df.loc[mask].copy()
            matched_df["model_name"] = f"{category}-matched PRS"
            matched_df["Model Name"] = f"{category}-matched PRS"

            sub_metrics_df = pd.concat(
                [matched_df.reset_index(drop=True), sub_metrics_df.reset_index(drop=True)],
                ignore_index=True,
            )

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
