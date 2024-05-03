import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


################### Helper functions ####################

def map_color(value, cmap_name='inferno', vmin=0., vmax=1.):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    return cmap(norm(abs(value)))

#########################################################


# Read the PGS data + metadata:
pgs_df = pd.read_csv("data/scores/EFO_0004339.csv.gz", sep="\t")
pgs_metadata = pd.read_csv("metadata/pgs_weights.txt")

#############################

# Plots (1): Show the correlation between the different polygenic scores.

# Generate the correlation matrix:
corr_mat = pgs_df.drop(['FID', 'IID'], axis=1).corr()

# Extract the correlation beween all the PGSs and the phenotype:

pheno_corr = corr_mat['height_rint'].drop('height_rint').to_frame()

# Map the correlation values to colors:
pheno_corr['height_rint'] = pheno_corr.index.map(
    dict(zip(pheno_corr.index, map_color(pheno_corr['height_rint'].values)))
)

# Extract the PGS-by-PGS correlation matrix (phenotype excluded):

final_corr_mat = corr_mat.drop('height_rint').drop('height_rint', axis=1)

# Translate the PGS IDs to PGS name for clarity and interpretability:

final_corr_mat.index = final_corr_mat.index.map(dict(zip(pgs_metadata['PGS_ID'], pgs_metadata['name'])))
pheno_corr.index = pheno_corr.index.map(dict(zip(pgs_metadata['PGS_ID'], pgs_metadata['name'])))

cg = sns.clustermap(final_corr_mat,
                    figsize=(15,12),
                    dendrogram_ratio=0.1,
                    yticklabels=True, xticklabels=True,
                    cbar_pos=(0., 0.5, 0.05, 0.18),
                    row_colors=pheno_corr,
                    cmap='inferno', vmin=0., vmax=1.)
cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)

plt.savefig("figures/exploratory/pgs_corr.svg")
plt.close()

#############################

# Plots (2): Show the distribution of polygenic scores for the UKB samples

plt.figure(figsize=(15, 8))
ndf = pgs_df.drop(['FID', 'IID'], axis=1)[list(cg.data2d.columns) + ['height_rint']]
ndf.columns = list(cg.data2d.index) + ['height_rint']
ndf.boxplot(showfliers=False)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("figures/exploratory/pgs_dist.svg")
plt.close()


