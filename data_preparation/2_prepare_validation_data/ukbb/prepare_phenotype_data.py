import os.path as osp
import sys

import numpy as np
import pandas as pd
from magenpy.utils.system_utils import makedir

sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import (
    adjust_diastolic_blood_pressure_for_medication,
    adjust_ldl_cholesterol_for_medication,
    adjust_systolic_blood_pressure_for_medication,
    adjust_total_cholesterol_for_medication,
    detect_outliers,
)

ukb_homedir = "/project/rpp-aevans-ab/neurohub/UKB/"
medication_use_file = "data/covariates/ukbb/medication_use.txt"

pheno_dict = {
    "48-0.0": "WAIST",
    "49-0.0": "HIP",
    "50-0.0": "HEIGHT",
    "21001-0.0": "LOG_BMI",
    "30760-0.0": "LOG_HDL",
    "30780-0.0": "LDL",
    "30690-0.0": "TC",
    "30870-0.0": "LOG_TG",
    # "20151-0.0": "FVC",
    # "20150-0.0": "FEV1",
    # "20258-0.0": "FEV1_FVC",
    "30700-0.0": "LOG_CRTN",
    "30880-0.0": "URT",
    "30850-0.0": "LOG_TST",
    "4080-0.0": "SBP",
    "4079-0.0": "DBP",
    # "6138-0.0": "EDU",
}

component_phenotypes = ["48-0.0", "49-0.0"]

# ------------------------------------------------------
# Helper functions to transform some of the phenotypes:


def transform_education_years(dat):
    """
    Transform educations level from categorical variable
    to education years defined by Okaby et al. 2016.
    Relevant tables are:

        https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id=100305
        https://elifesciences.org/articles/48376#app1table4
    """

    return dat.map({1: 20, 2: 13, 3: 10, 4: 10, 5: 19, 6: 15, -7: 7, -3: np.nan})


pheno_transform_func = {
    # "6138-0.0": transform_education_years,
    "30870-0.0": np.log,
    "21001-0.0": np.log,
    "30700-0.0": np.log,
    "30760-0.0": np.log,
    "30850-0.0": np.log,
}

pheno_adjust_func = {
    "30780-0.0": adjust_ldl_cholesterol_for_medication,
    "30690-0.0": adjust_total_cholesterol_for_medication,
    "4080-0.0": adjust_systolic_blood_pressure_for_medication,
    "4079-0.0": adjust_diastolic_blood_pressure_for_medication,
}

log_before_outlier_detection = ["20151-0.0", "20150-0.0"]


# ------------------------------------------------------
# Read quantitative phenotypes

pheno_df = pd.read_csv(
    osp.join(ukb_homedir, "Tabular/current.csv"),
    usecols=["eid", "22001-0.0"] + list(pheno_dict.keys()),
)
pheno_df.rename(columns={"eid": "IID", "22001-0.0": "Sex"}, inplace=True)


# Read the list of withdrawn individuals:
withdrawn_df = pd.read_csv(
    osp.join(ukb_homedir, "Withdrawals/w45551_20250818.csv"), names=["IID"]
)

# Read the medication use table:
med_use_df = pd.read_csv(medication_use_file, sep="\t")


# Remove withdrawn samples from df:
pheno_df = pheno_df[~pheno_df["IID"].isin(withdrawn_df["IID"])]
pheno_df["FID"] = pheno_df["IID"]

# Create the phenotype directory:
makedir("data/phenotypes/ukbb/")

# ------------------------------------------------------

# Loop over the phenotypes, process them, and output to file:
for pheno in pheno_dict.keys():
    if pheno in component_phenotypes:
        continue

    sub_pheno_df = pheno_df[["FID", "IID", pheno]].copy()
    sub_pheno_df.columns = ["FID", "IID", "phenotype"]

    # Apply phenotype-specific transforms:
    if pheno in pheno_transform_func:
        sub_pheno_df["phenotype"] = pheno_transform_func[pheno](
            sub_pheno_df["phenotype"]
        )

    # Filter outliers in each sex separately:
    # If the phenotype is skewed and positive,
    # apply log transformation before outlier detection.
    if pheno in log_before_outlier_detection:
        od_pheno = np.log(sub_pheno_df["phenotype"])
    else:
        od_pheno = sub_pheno_df["phenotype"]

    sub_pheno_df["phenotype"] = np.where(
        detect_outliers(od_pheno, stratify=pheno_df["Sex"]),
        np.nan,
        sub_pheno_df["phenotype"],
    )
    # Save the phenotype
    sub_pheno_df.to_csv(
        f"data/phenotypes/ukbb/{pheno_dict[pheno]}.txt",
        sep="\t",
        index=False,
        header=False,
        na_rep="NA",
    )

    # Adjust phenotype for medication use:
    if pheno in pheno_adjust_func:
        sub_pheno_df = pheno_adjust_func[pheno](sub_pheno_df, med_use_df)

        # Save the phenotype
        sub_pheno_df.to_csv(
            f"data/phenotypes/ukbb/{pheno_dict[pheno]}_adj.txt",
            sep="\t",
            index=False,
            header=False,
            na_rep="NA",
        )

# =============================================================================
# Computed phenotypes:

# 1) Waist-to-hip ratio (WHR):

sub_pheno_df = pheno_df[["FID", "IID", "48-0.0", "49-0.0"]].copy()
# Remove outliers in each separately:
sub_pheno_df["48-0.0"] = np.where(
    detect_outliers(sub_pheno_df["48-0.0"], stratify=pheno_df["Sex"]),
    np.nan,
    sub_pheno_df["48-0.0"],
)
sub_pheno_df["49-0.0"] = np.where(
    detect_outliers(sub_pheno_df["49-0.0"], stratify=pheno_df["Sex"]),
    np.nan,
    sub_pheno_df["49-0.0"],
)
# Compute WHR
sub_pheno_df["phenotype"] = sub_pheno_df["48-0.0"] / sub_pheno_df["49-0.0"]
# Remove outliers in WHR:
sub_pheno_df["phenotype"] = np.where(
    detect_outliers(sub_pheno_df["phenotype"], stratify=pheno_df["Sex"]),
    np.nan,
    sub_pheno_df["phenotype"],
)
# Save phenotype:
sub_pheno_df[["FID", "IID", "phenotype"]].to_csv(
    "data/phenotypes/ukbb/WHR.txt", sep="\t", index=False, header=False, na_rep="NA"
)
