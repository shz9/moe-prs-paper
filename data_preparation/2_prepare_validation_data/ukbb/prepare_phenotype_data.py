
import pandas as pd
import numpy as np
import os.path as osp
import sys
import itertools
from magenpy.utils.system_utils import makedir
sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import detect_outliers

ukb_homedir = "/lustre03/project/6004777/projects/uk_biobank/"

sample_data = pd.read_csv(osp.join(ukb_homedir, "lists/ukb_sqc_v2_fullID_head.txt"),
                          sep=r"\s+", usecols=['FID', 'IID', 'Inferred.Gender'])
sample_data.columns = ['FID', 'IID', 'Sex']
sample_ids = sample_data[['FID', 'IID']]

# ------------------------------------------------------
# Quantitative phenotypes

file_q_trait_dict = {
    "ukb5602.csv": ["50-0.0", "3062-0.0", "3063-0.0", "21001-0.0"],
    "ukb27843.csv": ["30760-0.0", "30780-0.0", "30690-0.0", "30870-0.0", "30700-0.0", "30880-0.0", "30850-0.0"]
}

# Construct the phenotype table:
pheno_df = None

for ph_file, traits in file_q_trait_dict.items():
    df = pd.read_csv(osp.join(ukb_homedir, 'raw', ph_file), usecols=["eid"] + traits)
    if pheno_df is None:
        pheno_df = df
    else:
        pheno_df = pheno_df.merge(df, on="eid")

pheno_df = sample_data.merge(pheno_df, left_on="IID", right_on="eid")
pheno_df.drop('eid', axis=1, inplace=True)

# Create the phenotype directory:
makedir("data/phenotypes/ukbb/")

# Standing height:

sh = pheno_df[['FID', 'IID', '50-0.0']]
sh.columns = ['FID', 'IID', 'phenotype']
sh['phenotype'] = np.where(detect_outliers(sh['phenotype'], stratify=pheno_df['Sex']), np.nan, sh['phenotype'])
sh.to_csv("data/phenotypes/ukbb/EFO_0004339.txt", sep="\t", index=False, header=False, na_rep='NA')

# BMI:

bmi = pheno_df[['FID', 'IID', '21001-0.0']]
bmi.columns = ['FID', 'IID', 'phenotype']
bmi['phenotype'] = np.where(detect_outliers(np.log(bmi['phenotype']), stratify=pheno_df['Sex']), np.nan, bmi['phenotype'])
bmi.to_csv("data/phenotypes/ukbb/EFO_0004340.txt", sep="\t", index=False, header=False, na_rep='NA')

# HDL:

hdl = pheno_df[['FID', 'IID', '30760-0.0']]
hdl.columns = ['FID', 'IID', 'phenotype']
hdl['phenotype'] = np.where(detect_outliers(np.log(hdl['phenotype']), stratify=pheno_df['Sex']), np.nan, hdl['phenotype'])
hdl.to_csv("data/phenotypes/ukbb/EFO_0004612.txt", sep="\t", index=False, header=False, na_rep='NA')

# LDL:

ldl = pheno_df[['FID', 'IID', '30780-0.0']]
ldl.columns = ['FID', 'IID', 'phenotype']
ldl['phenotype'] = np.where(detect_outliers(ldl['phenotype'], stratify=pheno_df['Sex']), np.nan, ldl['phenotype'])
ldl.to_csv("data/phenotypes/ukbb/EFO_0004611.txt", sep="\t", index=False, header=False, na_rep='NA')

# Total cholesterol:

total_chol = pheno_df[['FID', 'IID', '30690-0.0']]
total_chol.columns = ['FID', 'IID', 'phenotype']
total_chol['phenotype'] = np.where(detect_outliers(total_chol['phenotype'], stratify=pheno_df['Sex']), np.nan, total_chol['phenotype'])
total_chol.to_csv("data/phenotypes/ukbb/EFO_0004574.txt",
                  sep="\t", index=False, header=False, na_rep='NA')

# Log triglycerides:

log_trig = pheno_df[['FID', 'IID', '30870-0.0']]
log_trig.columns = ['FID', 'IID', 'phenotype']
log_trig['phenotype'] = np.log(log_trig['phenotype'])
log_trig['phenotype'] = np.where(detect_outliers(log_trig['phenotype'], stratify=pheno_df['Sex']), np.nan, log_trig['phenotype'])
log_trig.to_csv("data/phenotypes/ukbb/EFO_0004530.txt",
                sep="\t", index=False, header=False, na_rep='NA')

# FVC:

fvc = pheno_df[['FID', 'IID', '3062-0.0']]
fvc.columns = ['FID', 'IID', 'phenotype']
fvc['phenotype'] = np.where(detect_outliers(np.log(fvc['phenotype']), stratify=pheno_df['Sex']), np.nan, fvc['phenotype'])
# fvc.to_csv("data/phenotypes/ukbb/FVC.txt", sep="\t", index=False, header=False, na_rep='NA')

# FEV1

fev1 = pheno_df[['FID', 'IID', '3063-0.0']]
fev1.columns = ['FID', 'IID', 'phenotype']
fev1['phenotype'] = np.where(detect_outliers(np.log(fev1['phenotype']), stratify=pheno_df['Sex']), np.nan, fev1['phenotype'])
# fev1.to_csv("data/phenotypes/ukbb/FEV1.txt", sep="\t", index=False, header=False, na_rep='NA')

# FEV1/FVC:

fev1_fvc = fev1.merge(fvc, on=['FID', 'IID'], suffixes=('_fev1', '_fvc'))
fev1_fvc['phenotype'] = fev1_fvc['phenotype_fev1'] / fev1_fvc['phenotype_fvc']
fev1_fvc.drop(['phenotype_fev1', 'phenotype_fvc'], axis=1, inplace=True)
fev1_fvc['phenotype'] = np.where(detect_outliers(np.log(fev1_fvc['phenotype']), stratify=pheno_df['Sex']),
                                 np.nan, fev1_fvc['phenotype'])
fev1_fvc.to_csv("data/phenotypes/ukbb/EFO_0004713.txt", sep="\t", index=False, header=False, na_rep='NA')

# =============================================================================
# Process sex-biased phenotypes

# Creatinine

creatinine = pheno_df[['FID', 'IID', '30700-0.0']]
creatinine.columns = ['FID', 'IID', 'phenotype']
creatinine['phenotype'] = np.where(detect_outliers(creatinine['phenotype'], stratify=pheno_df['Sex']),
                                   np.nan, creatinine['phenotype'])
creatinine.to_csv("data/phenotypes/ukbb/EFO_0004518.txt",
                  sep="\t", index=False, header=False, na_rep='NA')

# Urate

urate = pheno_df[['FID', 'IID', '30880-0.0']]
urate.columns = ['FID', 'IID', 'phenotype']
urate['phenotype'] = np.where(detect_outliers(urate['phenotype'], stratify=pheno_df['Sex']),
                              np.nan, urate['phenotype'])
urate.to_csv("data/phenotypes/ukbb/EFO_0004531.txt",
             sep="\t", index=False, header=False, na_rep='NA')

# Testosterone

testosterone = pheno_df[['FID', 'IID', '30850-0.0']]
testosterone.columns = ['FID', 'IID', 'phenotype']
testosterone['phenotype'] = np.where(detect_outliers(testosterone['phenotype'], stratify=pheno_df['Sex']),
                                     np.nan, testosterone['phenotype'])
testosterone.to_csv("data/phenotypes/ukbb/EFO_0004908.txt",
                    sep="\t", index=False, header=False, na_rep='NA')

# =============================================================================
# Case/control phenotypes

icd10_file = "ukb5922.csv"
illness_file = "ukb4940.csv"

# Add ICD10 cause of death, primary + secondary
icd10_cols = [f"40001-{i}.0" for i in range(2)] + [f"40002-{i}.{j}" for i, j in itertools.product(range(2), range(14))]
# Add ICD10 diagnoses, main + secondary
icd10_cols += [f"41202-0.{i}" for i in range(377)] + [f"41204-0.{i}" for i in range(344)]
general_illness_cols = [f"20002-{i}.{j}" for i, j in itertools.product(range(2), range(29))]

# Read the files and merge them:
df_icd10 = pd.read_csv(osp.join(ukb_homedir, 'raw', icd10_file),
                       usecols=['eid'] + icd10_cols)
df_illness = pd.read_csv(osp.join(ukb_homedir, 'raw', illness_file),
                         usecols=['eid'] + general_illness_cols)

df_disease = df_icd10.merge(df_illness, on="eid")

# Merge on the covariate file:
df_disease = sample_ids.merge(df_disease, left_on="IID", right_on="eid")
df_disease.drop('eid', axis=1, inplace=True)

# ------------------ Asthma ------------------

# Extract index of individuals who have been diagnosed with asthma
asthma_idx = np.where(np.logical_or(
    (df_disease[general_illness_cols] == 1111).any(axis=1),
    df_disease[icd10_cols].applymap(lambda x: str(x)[:3] == "J45").any(axis=1)
))[0]

# Extract index of individuals who have asthma-related diagnoses (to be excluded)
asthma_like_idx = np.where(np.logical_or(
    df_disease[general_illness_cols].applymap(lambda x: x in list(range(1111, 1126))).any(axis=1),
    df_disease[icd10_cols].applymap(lambda x: str(x)[:1] == "J").any(axis=1)
))[0]

asthma_df = df_disease[['FID', 'IID']]
asthma_df['phenotype'] = 0
asthma_df.values[asthma_like_idx, -1] = -9
asthma_df.values[asthma_idx, -1] = 1

asthma_df = asthma_df.loc[asthma_df['phenotype'] != -9]
asthma_df.to_csv("data/phenotypes/ukbb/MONDO_0004979.txt", sep="\t", index=False, header=False, na_rep='NA')

# ------------------ T1D & T2D ------------------

# Extract index of individuals who have general diabetes diagnosis
diabetes_like_idx = np.where(np.logical_or(
    df_disease[general_illness_cols].applymap(lambda x: x in (1220, 1221, 1222, 1223)).any(axis=1),
    df_disease[icd10_cols].applymap(lambda x: str(x)[:3] in ("E10", "E11", "E12", "E13", "E14")).any(axis=1)
))[0]

# Extract index of individuals who have T1D diagnosis
t1d_idx = np.where(np.logical_or(
    (df_disease[general_illness_cols] == 1222).any(axis=1),
    df_disease[icd10_cols].applymap(lambda x: str(x)[:3] == "E10").any(axis=1)
))[0]

# Extract index of individuals who have T2D diagnosis
t2d_idx = np.where(np.logical_or(
    (df_disease[general_illness_cols] == 1223).any(axis=1),
    df_disease[icd10_cols].applymap(lambda x: str(x)[:3] == "E11").any(axis=1)
))[0]

# T1D:
t1d_df = df_disease[['FID', 'IID']]
t1d_df['phenotype'] = 0
t1d_df.values[diabetes_like_idx, -1] = -9
t1d_df.values[t1d_idx, -1] = 1
t1d_df.values[t2d_idx, -1] = -9

t1d_df = t1d_df.loc[t1d_df['phenotype'] != -9]
# t1d_df.to_csv("data/phenotypes/ukbb/T1D.txt", sep="\t", index=False, header=False, na_rep='NA')

# T2D:
t2d_df = df_disease[['FID', 'IID']]
t2d_df['phenotype'] = 0
t2d_df.values[diabetes_like_idx, -1] = -9
t2d_df.values[t2d_idx, -1] = 1
t2d_df.values[t1d_idx, -1] = -9

t2d_df = t2d_df.loc[t2d_df['phenotype'] != -9]
t2d_df.to_csv("data/phenotypes/ukbb/MONDO_0005148.txt", sep="\t", index=False, header=False, na_rep='NA')
