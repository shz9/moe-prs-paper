import pandas as pd
import seaborn as sns

merged_df = gnomad_df.merge(cluster_merged, on='IID')

import seaborn as sns
import matplotlib.pyplot as plt

# Assume `labels_A` and `labels_B` are your two label sets
conf_mat = pd.crosstab(merged_df.ancestry, merged_df.Description, normalize='columns')

# Plot the confusion matrix
plt.figure(figsize=(20, 8))
cg = sns.clustermap(conf_mat, fmt='.2f', annot=True, row_cluster=False, cmap='Blues', dendrogram_ratio=.05)
cg.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
cg.ax_col_dendrogram.set_visible(False) #suppress column dendrogram
cg.cax.set_visible(False)

plt.xlabel('UMAP Clusters')
plt.ylabel('gnomAD Labels')
plt.show()
