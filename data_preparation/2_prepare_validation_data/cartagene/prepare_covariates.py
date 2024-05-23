import pandas as pd
import numpy as np
import argparse
import os.path as osp
from magenpy.utils.system_utils import makedir


parser = argparse.ArgumentParser(description='Extract covariates from the Cartagene files')

parser.add_argument('--num-pcs', dest='num_pcs', type=int, default=10,
                    help='The number of PCs to extract into the covariates file.')
args = parser.parse_args()

cartagene_homedir = "$HOME/projects/ctb-sgravel/cartagene/research/quebec_structure_936028/"
cartagene_homedir = osp.expandvars(cartagene_homedir)

pc_df = pd.read_csv(osp.join(cartagene_homedir, "results/dimensionality_reduction/hla_removed_ld_thinned.eigenvec"),
                    header=None,
                    usecols=list(range(args.num_pcs + 2)),
                    delim_whitespace=True)

pc_df.columns = ['FID', 'IID'] + [f'PC{i+1}' for i in range(len(pc_df.columns) - 2)]

pheno_df = pd.read_csv(osp.join(cartagene_homedir, "data/old_metadata/data_Gravel936028_2.zip"),
                       usecols=['file111', 'p_age', 'sex_birth'])
pheno_df.columns = ['IID', 'Age', 'Sex']
pheno_df.dropna(inplace=True)
pheno_df['Sex'] = np.abs(pheno_df['Sex'] - 1)  # For consistency with other dataset, reverse coding of males/females

merged_df = pc_df.merge(pheno_df, on='IID')
merged_df = merged_df[['FID', 'IID', 'Sex'] + [f'PC{i+1}' for i in range(len(pc_df.columns) - 2)] + ['Age']]
merged_df['FID'] = 0

makedir("data/covariates/cartagene/")
merged_df.to_csv("data/covariates/cartagene/covars_cartagene_pcs.txt",
                 sep="\t", index=False, header=False)


# Extract cluster information from Alex's pipeline:

clust_df = pd.concat([
    pc_df.iloc[:, :2],
    pd.read_csv(osp.join(cartagene_homedir, "results/dimensionality_reduction/hdbscan_clusters/"
                                            "hdbscan_labels_min25_EPS0.3_hla_removed_ld_thinned.eigenvec_UMAP_"
                                            "PC25_NC3_NN10_MD0.001_euclidean_20220714_173646.txt"),
                names=['Cluster'])
], axis=1)

clust_df['FID'] = 0

clust_df.to_csv("data/covariates/cartagene/cluster_assignment.txt", sep="\t", index=False)