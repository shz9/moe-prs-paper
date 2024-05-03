import pandas as pd
import numpy as np
from magenpy.utils.system_utils import makedir
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from utils import detect_outliers


cols_dict = {
    'file111': 'IID',
    'sex_birth': 'Sex',
    'DIABETES_T2': 'T2D',
    'RES_MEASURED_FEV1': 'FEV1',
    'RES_MEASURED_FVC': 'FVC',
    'RES_BODY_MASS_INDEX': 'BMI',
    'CALC_AVG_HEIGHT_CM': 'HEIGHT',
    'ASTHMA_OCCURRENCE': 'ASTHMA'
}

# --------------------------------------------------------------------------------

pheno_df = pd.read_csv("../quebec_structure_936028/data/old_metadata/data_Gravel936028_2.zip",
                       usecols=list(cols_dict.keys()))

pheno_df.columns = [cols_dict[c] for c in pheno_df.columns]
pheno_df['FID'] = 0

# --------------------------------------------------------------------------------
# Quantitative phenotypes

makedir("data/phenotypes/cartagene/")

# Process data for height:

sh = pheno_df[['FID', 'IID', 'HEIGHT']].copy()
sh.columns = ['FID', 'IID', 'phenotype']
sh['phenotype'] = np.where(detect_outliers(sh['phenotype'], stratify=pheno_df['Sex']), np.nan, sh['phenotype'])
sh.to_csv("data/phenotypes/cartagene/EFO_0004339.txt", sep="\t", index=False, na_rep='NA')

# Process data for BMI:

bmi = pheno_df[['FID', 'IID', 'BMI']].copy()
bmi.columns = ['FID', 'IID', 'phenotype']
bmi['phenotype'] = np.where(detect_outliers(np.log(bmi['phenotype']), stratify=pheno_df['Sex']),
                            np.nan, bmi['phenotype'])
bmi.to_csv("data/phenotypes/cartagene/EFO_0004340.txt", sep="\t", index=False, na_rep='NA')


# Process data for FEV1:

fev1 = pheno_df[['FID', 'IID', 'FEV1']].copy()
fev1.columns = ['FID', 'IID', 'phenotype']
fev1['phenotype'][fev1['phenotype'] <= 0.] = np.nan
fev1['phenotype'] = np.where(detect_outliers(np.log(fev1['phenotype']), stratify=pheno_df['Sex']),
                             np.nan, fev1['phenotype'])
# fev1.to_csv("data/phenotypes/cartagene/FEV1.txt", sep="\t", index=False, na_rep='NA')

# Process data for FVC:

fvc = pheno_df[['FID', 'IID', 'FVC']].copy()
fvc.columns = ['FID', 'IID', 'phenotype']
fvc['phenotype'][fvc['phenotype'] <= 0.] = np.nan
fvc['phenotype'] = np.where(detect_outliers(np.log(fvc['phenotype']), stratify=pheno_df['Sex']),
                            np.nan, fvc['phenotype'])
# fev1.to_csv("data/phenotypes/cartagene/FEV1.txt", sep="\t", index=False, na_rep='NA')

# FEV1/FVC:

fev1_fvc = fev1.merge(fvc, on=['FID', 'IID'], suffixes=('_fev1', '_fvc'))
fev1_fvc['phenotype'] = fev1_fvc['phenotype_fev1'] / fev1_fvc['phenotype_fvc']
fev1_fvc.drop(['phenotype_fev1', 'phenotype_fvc'], axis=1, inplace=True)
fev1_fvc['phenotype'] = np.where(detect_outliers(np.log(fev1_fvc['phenotype']), stratify=pheno_df['Sex']),
                                 np.nan, fev1_fvc['phenotype'])
fev1_fvc.to_csv("data/phenotypes/cartagene/EFO_0004713.txt", sep="\t", index=False, header=False, na_rep='NA')

# --------------------------------------------------------------------------------
# Binary phenotypes

makedir("data/phenotypes/cartagene/")

# Process data for ASTHMA:

pheno_df = pheno_df[['FID', 'IID', 'ASTHMA']].copy()
pheno_df['ASTHMA'] = pheno_df['ASTHMA'].replace({2.: 0., 9.: np.nan, 99.: np.nan})
pheno_df.columns = ['FID', 'IID', 'phenotype']
pheno_df.to_csv("data/phenotypes/cartagene/MONDO_0004979.txt", sep="\t", index=False,  na_rep='NA')

# Process data for T2D:

pheno_df = pheno_df[['FID', 'IID', 'T2D']].copy()
pheno_df['T2D'] = pheno_df['T2D'].replace({2.: 0., 9.: np.nan, 99.:np.nan})
pheno_df.columns = ['FID', 'IID', 'phenotype']
pheno_df.to_csv("data/phenotypes/cartagene/MONDO_0005148.txt", sep="\t", index=False,  na_rep='NA')


# --------------------------------------------------------------------------------
# Blood biochemistry phenotypes

ind_df = pd.read_csv("../quebec_structure_936028/data/phenotypes/Gravel936028_5/Gravel_936028_5.genetic_codes.csv.gz")
pheno_df = pd.read_csv("../quebec_structure_936028/data/phenotypes/Gravel936028_5/Gravel_936028_5.mesures_biochimiques.csv.gz")


pheno_df = ind_df.merge(pheno_df, on='PROJECT_CODE')
pheno_df['FID'] = 0

pheno_df.rename(columns={'file_111': 'IID'}, inplace=True)

# Process data for LDL:

ldl = pheno_df[['FID', 'IID', 'LDL']].copy()
ldl.columns = ['FID', 'IID', 'phenotype']
ldl['phenotype'] = np.where(detect_outliers(ldl['phenotype'], stratify=pheno_df['Sex']),
                            np.nan, ldl['phenotype'])
ldl.to_csv("data/phenotypes/cartagene/EFO_0004611.txt", sep="\t", na_rep='NA')

# Process data for HDL:

hdl = pheno_df[['FID', 'IID', 'HDL']].copy()
hdl.columns = ['FID', 'IID', 'phenotype']
hdl['phenotype'] = np.where(detect_outliers(np.log(hdl['phenotype']), stratify=pheno_df['Sex']),
                            np.nan, hdl['phenotype'])
hdl.to_csv("data/phenotypes/cartagene/EFO_0004612.txt", sep="\t", na_rep='NA')

# Process data for Total Cholesterol:

total_chol = pheno_df[['FID', 'IID', 'TC']].copy()
total_chol.columns = ['FID', 'IID', 'phenotype']
total_chol['phenotype'] = np.where(detect_outliers(total_chol['phenotype'], stratify=pheno_df['Sex']),
                                   np.nan, total_chol['phenotype'])
total_chol.to_csv("data/phenotypes/cartagene/EFO_0004574.txt", sep="\t", na_rep='NA')

# Process data for triglycerides:
log_trig = pheno_df[['FID', 'IID', 'TRIG']].copy()
log_trig.columns = ['FID', 'IID', 'phenotype']
log_trig['phenotype'] = np.log(log_trig['phenotype'])
log_trig['phenotype'] = np.where(detect_outliers(log_trig['phenotype'], stratify=pheno_df['Sex']),
                                 np.nan, log_trig['phenotype'])
log_trig.to_csv("data/phenotypes/cartagene/EFO_0004530.txt", sep="\t", na_rep='NA')

# Process data for Creatinine:

creatinine = pheno_df[['FID', 'IID', 'CREATININE']].copy()
creatinine.columns = ['FID', 'IID', 'phenotype']
creatinine['phenotype'] = np.where(detect_outliers(creatinine['phenotype'], stratify=pheno_df['Sex']),
                                   np.nan, creatinine['phenotype'])
creatinine.to_csv("data/phenotypes/cartagene/EFO_0004518.txt", sep="\t", na_rep='NA')

