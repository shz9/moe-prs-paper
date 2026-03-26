import os.path as osp
import sys

import pandas as pd

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
    "cartagene": "CARTaGENE",
}

BIOBANK_NAME_MAP_SHORT = {
    "ukbb": "UKB",
    "cartagene": "CaG",
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
    import seaborn as sns

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

    remaining_models = sorted(list(set(MODEL_NAME_MAP.values()) - set(colors.keys())))
    remaining_colors = sns.color_palette(palette, len(remaining_models))

    colors.update(dict(zip(remaining_models, remaining_colors)))

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
    eval_df["Test biobank"] = file_path.split("/")[-2].upper()

    return eval_df


def transform_eval_metrics(eval_df):
    def map_model_name(x):
        try:
            biobank, rest = x.split("/")
            _, rest = rest.split(":")

            if rest in (pd.Series(MODEL_NAME_MAP.keys()) + "-covariates").values:
                m = rest.replace("-covariates", "")
            else:
                m = rest

            try:
                return MODEL_NAME_MAP[m] + f" ({biobank})"
            except KeyError as e:
                return m + f" ({biobank})"
        except ValueError as e:
            try:
                return MODEL_NAME_MAP[x]
            except Exception as e:
                return x

    def assign_training_cohort(x):
        try:
            biobank, rest = x.split("/")
            _, rest = rest.split(":")

            if rest in (pd.Series(MODEL_NAME_MAP.keys()) + "-covariates").values:
                m = rest.replace("-covariates", "")
            else:
                m = rest

            try:
                return MODEL_NAME_MAP[m]
            except KeyError as e:
                return m
        except ValueError as e:
            return x

    def map_dataset_name(x):
        try:
            split_x = x.split(":")
            if len(split_x) > 1:
                return split_x[0]
            else:
                return None
        except Exception as e:
            return x

    def assign_training_biobank(x):
        try:
            split_x = x.split("/")
            if len(split_x) > 1:
                return split_x[0].upper()
            else:
                return None
        except Exception as e:
            return x

    def assign_model_category(x):
        if "MoE" in x:
            return "MoE"
        elif "MultiPRS" in x:
            return "MultiPRS"
        elif "AncestryWeightedPRS" in x:
            return "AncestryWeightedPRS"
        elif "Covariates" in x:
            return "Covariates"
        elif "Random" in x:
            return "Random"
        else:
            if "covariates" in x:
                return "SinglePRS+Covariates"
            else:
                return "SinglePRS"

    eval_df["Training biobank"] = eval_df["PGS"].apply(assign_training_biobank)
    eval_df["Training dataset"] = eval_df["PGS"].apply(map_dataset_name)
    eval_df["Training cohort"] = eval_df["PGS"].apply(assign_training_cohort)
    eval_df["Model Name"] = eval_df["PGS"].apply(map_model_name)
    eval_df["Model Category"] = eval_df["PGS"].apply(assign_model_category)

    def map_group_name(x):
        try:
            return GROUP_MAP[x]
        except Exception:
            return x

    eval_df["EvalGroup"] = eval_df["EvalGroup"].astype(str).apply(map_group_name)
    eval_df.rename(columns={"EvalGroup": "Evaluation Group"}, inplace=True)

    return eval_df
