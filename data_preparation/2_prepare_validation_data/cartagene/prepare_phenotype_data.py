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

cols_dict = {
    "file111": "IID",
    "sex_birth": "Sex",
    #'RES_MEASURED_FEV1': 'FEV1',
    #'RES_MEASURED_FVC': 'FVC',
    "RES_BODY_MASS_INDEX": "BMI",
    "CALC_AVG_HEIGHT_CM": "HEIGHT",
    "ASTHMA_OCCURRENCE": "ASTHMA",
    "DIABETES_T1": "T1D",
    "DIABETES_T2": "T2D",
    # CAD related phenotypes:
    "HEART_ATTACK_OCCURRENCE": "CAD1",
    "ANGINA_OCCURRENCE": "CAD2",
    "ATHEROSCLEROSIS_OCCURRENCE": "CAD3",
    "HEART_DISEASE_OCCURRENCE": "CAD4",
    # Stroke:
    "STROKE_OCCURRENCE": "STR",
    # Hypertension:
    "HIGH_BP_OCCURRENCE": "HTN",
    # Gout:
    "GOUT_OCCURRENCE": "GOUT",
    #'HIGHEST_LVL_COMPLETED': 'EDU',
    "CALC_AVG_DIASTOLIC_BP": "DBP",
    "CALC_AVG_SYSTOLIC_BP": "SBP",
    "CALC_WAIST_TO_HIP_RATIO": "WHR",
}

cartagene_homedir = "$HOME/links/projects/def-sgravel/cartagene/"
cartagene_homedir = osp.expandvars(cartagene_homedir)

medication_use_file = "data/covariates/cartagene/medication_use.txt"

# --------------------------------------------------------------------------------

pheno_df = pd.read_csv(
    osp.join(cartagene_homedir, "old_metadata/data_Gravel936028_2.zip"),
    usecols=list(cols_dict.keys()),
)

pheno_df.columns = [cols_dict[c] for c in pheno_df.columns]
pheno_df["FID"] = pheno_df["IID"]

# Read the medication use table:
med_use_df = pd.read_csv(medication_use_file, sep="\t")

# --------------------------------------------------------------------------------
# Quantitative phenotypes


def transform_education_years(dat):
    """
    This function transforms the "HIGHEST_LVL_COMPLETED" variable
    in the CARTaGENE biobank to "years of education" as defined in the
    UK Biobank and similar resources.
    """

    return dat.map(
        {
            -9: np.nan,  # Not available
            -7: np.nan,  # Not applicable
            1: 7,  # NONE
            2: 7,  # ELEMENTARY
            3: 10,  # HIGH SCHOOL
            4: 15,  # TECHNICAL SCHOOL
            5: 19,  # COLLEGE
            6: 19,  # UNIVERSITY CERTIFICATE
            7: 20,  # BACHELOR DEGREE
            8: 20,  # GRADUATE STUDIES
            9: np.nan,  # NO ANSWER
            99: np.nan,  # MISSING
        }
    )


def recode_nan(df, col="phenotype"):
    """
    Recode the missing values in the dataframe to NaN.
    """
    df[col] = df[col].replace({-9: np.nan, 999: np.nan})
    return df


makedir("data/phenotypes/cartagene/")

# -------------------------------------------------------

# Process data for height:

sh = pheno_df[["FID", "IID", "HEIGHT"]].copy()
sh.columns = ["FID", "IID", "phenotype"]
# Remove samples with likely clerical errors:
sh.loc[(sh["phenotype"] < 130.0) | (sh["phenotype"] > 220), "phenotype"] = np.nan
# Recode NaN values:
sh = recode_nan(sh)
# Remove outliers per sex:
sh["phenotype"] = np.where(
    detect_outliers(sh["phenotype"], stratify=pheno_df["Sex"]), np.nan, sh["phenotype"]
)
print("Standing height")
print(sh["phenotype"].describe())
sh.to_csv(
    "data/phenotypes/cartagene/HEIGHT.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------

# Process data for BMI:

bmi = pheno_df[["FID", "IID", "BMI"]].copy()
bmi.columns = ["FID", "IID", "phenotype"]
# Remove samples with likely clerical errors:
bmi.loc[(bmi["phenotype"] < 15) | (bmi["phenotype"] > 60), "phenotype"] = np.nan
bmi = recode_nan(bmi)
bmi["phenotype"] = np.log(bmi["phenotype"])
bmi["phenotype"] = np.where(
    detect_outliers(bmi["phenotype"], stratify=pheno_df["Sex"]),
    np.nan,
    bmi["phenotype"],
)
print("Body Mass Index")
print(bmi["phenotype"].describe())
bmi.to_csv(
    "data/phenotypes/cartagene/LOG_BMI.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------

"""
# Process data for FEV1:
fev1 = pheno_df[["FID", "IID", "FEV1"]].copy()
fev1.columns = ["FID", "IID", "phenotype"]
fev1 = recode_nan(fev1)
fev1["phenotype"][fev1["phenotype"] <= 0.0] = np.nan
fev1["phenotype"] = np.where(
    detect_outliers(np.log(fev1["phenotype"]), stratify=pheno_df["Sex"]),
    np.nan,
    fev1["phenotype"],
)
# fev1.to_csv("data/phenotypes/cartagene/FEV1.txt", sep="\t", index=False, na_rep='NA')

# Process data for FVC:

fvc = pheno_df[["FID", "IID", "FVC"]].copy()
fvc.columns = ["FID", "IID", "phenotype"]
fvc = recode_nan(fvc)
fvc["phenotype"][fvc["phenotype"] <= 0.0] = np.nan
fvc["phenotype"] = np.where(
    detect_outliers(np.log(fvc["phenotype"]), stratify=pheno_df["Sex"]),
    np.nan,
    fvc["phenotype"],
)
# fev1.to_csv("data/phenotypes/cartagene/FEV1.txt", sep="\t", index=False, na_rep='NA')

# FEV1/FVC:

fev1_fvc = fev1.merge(fvc, on=["FID", "IID"], suffixes=("_fev1", "_fvc"))
fev1_fvc["phenotype"] = fev1_fvc["phenotype_fev1"] / fev1_fvc["phenotype_fvc"]
fev1_fvc.drop(["phenotype_fev1", "phenotype_fvc"], axis=1, inplace=True)
fev1_fvc["phenotype"] = np.where(
    detect_outliers(np.log(fev1_fvc["phenotype"]), stratify=pheno_df["Sex"]),
    np.nan,
    fev1_fvc["phenotype"],
)
print("FEV1/FVC")
print(fev1_fvc["phenotype"].describe())
print(pd.Series(zscore(fev1_fvc["phenotype"], nan_policy="omit")).describe())
fev1_fvc.to_csv(
    "data/phenotypes/cartagene/FEV1_FVC.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)
"""

# -------------------------------------------------------

# DBP:

dbp = pheno_df[["FID", "IID", "DBP"]].copy()
dbp.columns = ["FID", "IID", "phenotype"]
# Remove samples with likely clerical errors:
dbp.loc[(dbp["phenotype"] < 40) | (dbp["phenotype"] > 150), "phenotype"] = np.nan
dbp = recode_nan(dbp)
dbp["phenotype"] = np.where(
    detect_outliers(dbp["phenotype"], stratify=pheno_df["Sex"]),
    np.nan,
    dbp["phenotype"],
)
print("Diastolic blood pressure")
print(dbp["phenotype"].describe())
dbp.to_csv(
    "data/phenotypes/cartagene/DBP.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# Adjust for medication use:
dbp = adjust_diastolic_blood_pressure_for_medication(dbp, med_use_df)
dbp.to_csv(
    "data/phenotypes/cartagene/DBP_adj.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------

# SBP:
sbp = pheno_df[["FID", "IID", "SBP"]].copy()
sbp.columns = ["FID", "IID", "phenotype"]
# Remove samples with likely clerical errors:
sbp.loc[(sbp["phenotype"] < 70) | (sbp["phenotype"] > 250), "phenotype"] = np.nan
sbp = recode_nan(sbp)
sbp["phenotype"] = np.where(
    detect_outliers(sbp["phenotype"], stratify=pheno_df["Sex"]),
    np.nan,
    sbp["phenotype"],
)
print("Systolic blood pressure")
print(sbp["phenotype"].describe())
sbp.to_csv(
    "data/phenotypes/cartagene/SBP.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# Adjust for medication use:
sbp = adjust_systolic_blood_pressure_for_medication(sbp, med_use_df)
sbp.to_csv(
    "data/phenotypes/cartagene/SBP_adj.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------

# WHR:
whr = pheno_df[["FID", "IID", "WHR"]].copy()
whr.columns = ["FID", "IID", "phenotype"]
# Remove samples with likely clerical errors:
whr.loc[(whr["phenotype"] < 0.5) | (whr["phenotype"] > 1.5), "phenotype"] = np.nan
whr = recode_nan(whr)
whr["phenotype"] = whr["phenotype"].replace({9: np.nan})
whr["phenotype"] = np.where(
    detect_outliers(whr["phenotype"], stratify=pheno_df["Sex"]),
    np.nan,
    whr["phenotype"],
)
print("Waist-to-hip ratio")
print(whr["phenotype"].describe())
whr.to_csv(
    "data/phenotypes/cartagene/WHR.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# --------------------------------------------------------------------------------
# Binary phenotypes

binary_nan = {2.0: 0.0, 9.0: np.nan, 99.0: np.nan, -7.0: np.nan, -9.0: np.nan}

# Process data for ASTHMA:

asthma = pheno_df[["FID", "IID", "ASTHMA"]].copy()
asthma["ASTHMA"] = asthma["ASTHMA"].replace(binary_nan)
asthma.columns = ["FID", "IID", "phenotype"]
print("Asthma")
print(asthma["phenotype"].describe())
asthma.to_csv(
    "data/phenotypes/cartagene/ASTHMA.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------
# Process data for stroke:

stroke = pheno_df[["FID", "IID", "STR"]].copy()
stroke["STR"] = stroke["STR"].replace(binary_nan)
stroke.columns = ["FID", "IID", "phenotype"]
print("Stroke")
print(stroke["phenotype"].describe())
stroke.to_csv(
    "data/phenotypes/cartagene/STR.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------
# Process data for T2D:

t2d = pheno_df[["FID", "IID", "T2D"]].copy()
t2d["T2D"] = t2d["T2D"].replace(binary_nan)
t2d.columns = ["FID", "IID", "phenotype"]
print("T2D")
print(t2d["phenotype"].describe())
t2d.to_csv(
    "data/phenotypes/cartagene/T2D.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------
# Process data for T1D:

t1d = pheno_df[["FID", "IID", "T1D"]].copy()
t1d["T1D"] = t1d["T1D"].replace(binary_nan)
t1d.columns = ["FID", "IID", "phenotype"]
print("T1D")
print(t1d["phenotype"].describe())
t1d.to_csv(
    "data/phenotypes/cartagene/T1D.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------
# Process data for HTN:

htn = pheno_df[["FID", "IID", "HTN"]].copy()
htn["HTN"] = htn["HTN"].replace(binary_nan)
htn.columns = ["FID", "IID", "phenotype"]
print("HTN")
print(htn["phenotype"].describe())
htn.to_csv(
    "data/phenotypes/cartagene/HTN.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------
# Process data for GOUT:

gout = pheno_df[["FID", "IID", "GOUT"]].copy()
gout["GOUT"] = gout["GOUT"].replace(binary_nan)
gout.columns = ["FID", "IID", "phenotype"]
print("GOUT")
print(gout["phenotype"].describe())
gout.to_csv(
    "data/phenotypes/cartagene/GOUT.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------
# Process data for CAD:

cad = pheno_df[["FID", "IID", "CAD1", "CAD2", "CAD3", "CAD4"]].copy()
cad[["CAD1", "CAD2", "CAD3", "CAD4"]] = cad[["CAD1", "CAD2", "CAD3", "CAD4"]].replace(
    binary_nan
)
cad["CAD"] = (
    cad[["CAD1", "CAD2", "CAD3", "CAD4"]]
    .fillna(0.0)
    .sum(axis=1)
    .clip(lower=0.0, upper=1.0)
)
cad.drop(columns=["CAD1", "CAD2", "CAD3", "CAD4"], inplace=True)
cad.columns = ["FID", "IID", "phenotype"]
print("CAD")
print(cad["phenotype"].describe())
cad.to_csv(
    "data/phenotypes/cartagene/CAD.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# --------------------------------------------------------------------------------
# Blood biochemistry phenotypes

ind_df = pd.read_csv(
    osp.join(
        cartagene_homedir,
        "phenotypes/Gravel936028_5/Gravel_936028_5.genetic_codes.csv.gz",
    )
)
blood_pheno_df = pd.read_csv(
    osp.join(
        cartagene_homedir,
        "phenotypes/Gravel936028_5/Gravel_936028_5.mesures_biochimiques.csv.gz",
    )
)


blood_pheno_df = ind_df.merge(blood_pheno_df, on="PROJECT_CODE")
blood_pheno_df.rename(columns={"file_111": "IID"}, inplace=True)
blood_pheno_df["FID"] = blood_pheno_df["IID"]

blood_pheno_df = blood_pheno_df.merge(pheno_df[["IID", "Sex"]])

# -------------------------------------------------------
# Process data for LDL:

ldl = blood_pheno_df[["FID", "IID", "LDL"]].copy()
ldl.columns = ["FID", "IID", "phenotype"]
ldl = recode_nan(ldl)
ldl["phenotype"] = np.where(
    detect_outliers(ldl["phenotype"], stratify=blood_pheno_df["Sex"]),
    np.nan,
    ldl["phenotype"],
)
print("LDL")
print(ldl["phenotype"].describe())
ldl.to_csv(
    "data/phenotypes/cartagene/LDL.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# Adjust for medication use:
ldl = adjust_ldl_cholesterol_for_medication(ldl, med_use_df)
ldl.to_csv(
    "data/phenotypes/cartagene/LDL_adj.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------
# Process data for HDL:

hdl = blood_pheno_df[["FID", "IID", "HDL"]].copy()
hdl.columns = ["FID", "IID", "phenotype"]
hdl = recode_nan(hdl)
hdl["phenotype"] = np.log(hdl["phenotype"])
hdl["phenotype"] = np.where(
    detect_outliers(hdl["phenotype"], stratify=blood_pheno_df["Sex"]),
    np.nan,
    hdl["phenotype"],
)
print("LOG_HDL")
print(hdl["phenotype"].describe())
hdl.to_csv(
    "data/phenotypes/cartagene/LOG_HDL.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------
# Process data for Total Cholesterol:

total_chol = blood_pheno_df[["FID", "IID", "TC"]].copy()
total_chol.columns = ["FID", "IID", "phenotype"]
total_chol = recode_nan(total_chol)
total_chol["phenotype"] = np.where(
    detect_outliers(total_chol["phenotype"], stratify=blood_pheno_df["Sex"]),
    np.nan,
    total_chol["phenotype"],
)
print("Total Cholesterol")
print(total_chol["phenotype"].describe())
total_chol.to_csv(
    "data/phenotypes/cartagene/TC.txt", sep="\t", index=False, header=False, na_rep="NA"
)

# Adjust for medication use:
total_chol = adjust_total_cholesterol_for_medication(total_chol, med_use_df)
total_chol.to_csv(
    "data/phenotypes/cartagene/TC_adj.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------

# Process data for triglycerides:
log_trig = blood_pheno_df[["FID", "IID", "TRIG"]].copy()
log_trig.columns = ["FID", "IID", "phenotype"]
log_trig = recode_nan(log_trig)
log_trig["phenotype"] = np.log(log_trig["phenotype"])
log_trig["phenotype"] = np.where(
    detect_outliers(log_trig["phenotype"], stratify=blood_pheno_df["Sex"]),
    np.nan,
    log_trig["phenotype"],
)
print("Log Triglycerides")
print(log_trig["phenotype"].describe())
log_trig.to_csv(
    "data/phenotypes/cartagene/LOG_TG.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------
# Process data for Creatinine:

creatinine = blood_pheno_df[["FID", "IID", "CREATININE"]].copy()
creatinine.columns = ["FID", "IID", "phenotype"]
creatinine = recode_nan(creatinine)
creatinine["phenotype"] = np.log(creatinine["phenotype"])
creatinine["phenotype"] = np.where(
    detect_outliers(creatinine["phenotype"], stratify=blood_pheno_df["Sex"]),
    np.nan,
    creatinine["phenotype"],
)
print("Log Creatinine")
print(creatinine["phenotype"].describe())
creatinine.to_csv(
    "data/phenotypes/cartagene/LOG_CRTN.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)

# -------------------------------------------------------
# Process data for Urate:

urate = blood_pheno_df[["FID", "IID", "uric_acid"]].copy()
urate.columns = ["FID", "IID", "phenotype"]
urate = recode_nan(urate)
urate["phenotype"] = np.where(
    detect_outliers(urate["phenotype"], stratify=blood_pheno_df["Sex"]),
    np.nan,
    urate["phenotype"],
)
print("Urate")
print(urate["phenotype"].describe())
urate.to_csv(
    "data/phenotypes/cartagene/URT.txt",
    sep="\t",
    index=False,
    header=False,
    na_rep="NA",
)
