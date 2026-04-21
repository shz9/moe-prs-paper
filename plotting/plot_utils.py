import os.path as osp
import sys

import pandas as pd
import seaborn as sns

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "model/"))

from model_utils import get_analysis_id_mapper, get_model_name_mapper

# --------------------------------------------------------


MODEL_NAME_MAP = get_model_name_mapper()
ANALYSIS_TO_PHENOTYPE_MAP = get_analysis_id_mapper(target_col="Phenotype")
ANALYSIS_TO_SHORT_PHENOTYPE_MAP = get_analysis_id_mapper(target_col="Phenotype_short")

GROUP_MAP = {"0": "Female", "1": "Male"}

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
    "STR": "#8DD3C7",
    "STR_433": "#FFED6F",
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


def read_eval_metrics(file_path):
    """
    Read the evaluation metrics from a CSV file and transform the names
    of the models + the phenotype for the purposes of plotting.
    """

    eval_df = pd.read_csv(file_path)
    analysis_id = file_path.split("/")[-3]

    eval_df["AnalysisID"] = analysis_id
    eval_df["Phenotype"] = ANALYSIS_TO_PHENOTYPE_MAP[analysis_id]
    eval_df["Test biobank"] = file_path.split("/")[-2].upper()

    return eval_df


def transform_eval_metrics(eval_df):
    # ----------------------------------------------------------------------
    # Extract details about the training cohort / PGS:
    def process_pgs(x, analysis_id):
        NEW_MAP = MODEL_NAME_MAP[analysis_id]

        result = {
            "Training biobank": None,
            "Training dataset": None,
            "Training cohort": None,
            "Model Name": x,
            "Model Category": None,
        }

        # Parse biobank and model key from "biobank/dataset:model" format
        try:
            split_slash = x.split("/")
            if len(split_slash) > 1:
                biobank = split_slash[0]
                result["Training biobank"] = biobank.upper()
                rest = split_slash[1]
            else:
                biobank = None
                rest = x

            split_colon = rest.split(":")
            if len(split_colon) > 1:
                result["Training dataset"] = split_colon[0]
                m_raw = split_colon[1]
            else:
                m_raw = rest

            # Strip "-covariates" suffix to get the base model key
            if m_raw in (pd.Series(NEW_MAP.keys()) + "-covariates").values:
                m = m_raw.replace("-covariates", "")
            else:
                m = m_raw

            # Resolve model name from map
            mapped_name = NEW_MAP.get(m, m)
            result["Training cohort"] = mapped_name
            result["Model Name"] = mapped_name + (f" ({biobank})" if biobank else "")

        except (ValueError, AttributeError):
            result["Model Name"] = NEW_MAP.get(x, x)
            result["Training cohort"] = NEW_MAP.get(x, x)

        # Assign model category from the resolved model name
        model_name = result["Model Name"]
        if "MoE" in model_name:
            result["Model Category"] = "MoE"
        elif "MultiPRS" in model_name:
            result["Model Category"] = "MultiPRS"
        elif "AncestryWeightedPRS" in model_name:
            result["Model Category"] = "AncestryWeightedPRS"
        elif "SexMatchedPRS" in model_name:
            result["Model Category"] = "AttributePartitionedPRS"
        elif "Covariates" in model_name:
            result["Model Category"] = "Covariates"
        elif "Random" in model_name:
            result["Model Category"] = "Random"
        elif "covariates" in model_name:
            result["Model Category"] = "SinglePRS+Covariates"
        else:
            result["Model Category"] = "SinglePRS"

        return pd.Series(result)

    eval_df[
        [
            "Training biobank",
            "Training dataset",
            "Training cohort",
            "Model Name",
            "Model Category",
        ]
    ] = eval_df.apply(lambda row: process_pgs(row["PGS"], row["AnalysisID"]), axis=1)

    # ----------------------------------------------------------------------
    # Clean up the names of the evaluation cohorts:

    def map_group_name(x):
        try:
            return GROUP_MAP[x]
        except Exception:
            return x

    eval_df["EvalGroup"] = eval_df["EvalGroup"].astype(str).apply(map_group_name)
    eval_df.rename(columns={"EvalGroup": "Evaluation Group"}, inplace=True)

    return eval_df
