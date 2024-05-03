import sys
import os.path as osp
from magenpy.utils.system_utils import makedir
import pandas as pd
import functools
print = functools.partial(print, flush=True)


num_pcs = 10

# File names:
covar_file = "data/covariates/ukbb/covars_ukbb_pcs.txt"
keep_file = "data/keep_files/ukbb_qc_individuals.keep"

# -------- Sample quality control --------
# Read the sample QC file from the UKBB archive
print("> Extracting individuals data...")

ind_list = pd.read_csv("/lustre03/project/6004777/projects/uk_biobank/lists/ukb_sqc_v2_fullID_head.txt",
                       sep=r"\s+")

# Apply the standard filters:

ind_list = ind_list.loc[(ind_list['IID'] > 0) &  # Remove redacted samples
                        (ind_list['used.in.pca.calculation'] == 1) &  # Keep samples used in PCA calculation
                        (ind_list['excess.relatives'] == 0) &  # Remove samples with excess relatives
                        (ind_list['putative.sex.chromosome.aneuploidy'] == 0),  # Remove samples with sex chr aneuploidy
                        ['FID', 'IID', 'Inferred.Gender'] +
                        [f'PC{i+1}' for i in range(num_pcs)]]


# Write the list of remaining individuals to file:
makedir(osp.dirname(keep_file))
ind_list[['FID', 'IID']].to_csv(keep_file, sep="\t", header=False, index=False)

# -------- Sample covariate file --------
# Create a covariates file to use in GWAS:

print("Creating a file with covariates for the selected individuals...")

pc_columns = [f'PC{i+1}' for i in range(num_pcs)]

# Need this file to get the age of the participant:
ind_data = pd.read_csv("/lustre03/project/6004777/projects/uk_biobank/raw/ukb4940.csv")
ind_data = ind_data[['eid', '21003-0.0']]
ind_data.columns = ['IID', 'age']

covar_df = pd.merge(ind_list[['FID', 'IID', 'Inferred.Gender'] + pc_columns],
                    ind_data)
# Fix the representation for a couple of columns:
covar_df['Inferred.Gender'] = covar_df['Inferred.Gender'].map({'M': 1, 'F': 0})

for col in pc_columns:
    covar_df[col] = covar_df[col].round(decimals=5)

# Write the file:
makedir(osp.dirname(covar_file))
covar_df.to_csv(covar_file, sep="\t", header=False, index=False)