import sys
import os.path as osp
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(parent_dir, "model/"))
import pandas as pd
import numpy as np
from PRSDataset import PRSDataset


def generate_predictions(prs_dataset, models):

    preds = {}

    for m_name, m in models.items():
        preds[m_name] = m.predict(prs_dataset).flatten()

    return pd.DataFrame(preds)


def generate_pc_cluster_masks(prs_dataset, reference='median', n_clusters=5):
    """
    Cluster samples based on their distance in Principal Component space from
    a reference point (mean or median). This function takes a PRSDataset object
    and returns a dictionary of masks, where each key is the quantile distance
    index and the value is a boolean mask for the samples in that quantile.

    :param prs_dataset: A PRSDataset object.
    :param reference: The reference point to use for the distance calculation.
                      Can be either 'median' or 'mean'.
    :param n_clusters: The number of clusters to use for the quantile distance calculation.

    """

    masks = {'PC_DIST': {}}

    pc_dist_clust = rank_individuals_by_pc_distance(prs_dataset, reference, n_clusters=n_clusters)

    for pc_clust in np.unique(pc_dist_clust):
        masks['PC_DIST'][pc_clust] = pc_dist_clust == pc_clust

    return masks


def generate_continuous_masks(prs_dataset, cont_group_cols, n_bins=4):
    """
    Generate masks based on the quantiles of continuous columns in the
    PRS dataset. This function takes a PRSDataset object and a list of
    continuous columns by which to group the samples. It returns a nested
    dictionary of masks, where each key is a group name and the value is a
    dictionary of masks for that group. For example, if Age is used as a
    continuous variable, we find the appropriate quantiles based on
    the number of bins and return the following dictionary:

    {
        'Age': {
            'Age (Q1)': age == q1_age,
            'Age (Q2)': age == q2_age,
            'Age (Q3)': age == q3_age,
            'Age (Q4)': age == q4_age
        }
    }

    :param prs_dataset: A PRSDataset object
    :param cont_group_cols: A list of continuous columns by which to group the samples
    :param n_bins: The number of bins to use for the quantiles. Can be an integer or a list of
    integers the same length as cont_group_cols.

    """

    prs_dataset.set_backend("numpy")

    if isinstance(n_bins, int):
        n_bins = [n_bins]*len(cont_group_cols)

    if isinstance(cont_group_cols, str):
        cont_group_cols = [cont_group_cols]

    masks = {}

    for gcol, gbins in zip(cont_group_cols, n_bins):

        col_data = prs_dataset.get_data_columns(gcol).flatten()

        masks[gcol] = {}

        try:
            qcut_groups = pd.qcut(col_data, gbins, labels=list(range(gbins)))
            for i in range(gbins):
                masks[gcol][f"{gcol} (Q{i+1})"] = qcut_groups == i
        except ValueError as e:
            print(e)
            continue

    return masks


def generate_categorical_masks(prs_dataset, cat_group_cols, min_group_size=30):
    """
    Generate masks for the different groups in the dataset.
    This function takes a PRSDataset object and a list of categorical columns
    by which to group the samples. It returns a nested dictionary of masks, where each
    key is a group name and the value is a dictionary of masks for that group.
    For example, if Sex is used as a categorical variable to stratify the samples,
    the output would be:

    {
        'Sex': {
            'Males': mask
            'Females': ~mask
        }
    }

    :param prs_dataset: A PRSDataset object
    :param cat_group_cols: A list of categorical columns by which to group the samples

    """

    prs_dataset.set_backend("numpy")

    if isinstance(cat_group_cols, str):
        cat_group_cols = [cat_group_cols]

    masks = {}

    for gcol in cat_group_cols:

        # Get the data for the group column:
        col_data = prs_dataset.get_data_columns(gcol).flatten()

        # Determine the unique categories in the column:
        uniq_cats = np.unique(col_data)

        # If the categorical variable contains a single category, skip it
        if len(uniq_cats) < 2:
            print("> Skipping", gcol, "as it contains a single category in this dataset.")
            continue

        # Initialize the masks dictionary for the group column:
        masks[gcol] = {}

        for cat in np.unique(col_data):
            msk = col_data == cat
            if msk.sum() < min_group_size:
                print(f"> Skipping {gcol}={cat} as it contains less than {min_group_size} samples.")
                continue

            # If the category is a numeric value, first check that it can be converted
            # to an integer and then convert to a string:
            try:
                if float(cat) == int(cat):
                    cat = str(int(cat))
            except ValueError:
                pass

            masks[gcol][cat] = msk

    return masks


def rank_groups_by_pc_distance(prs_dataset,
                               group_col,
                               reference_group="largest"):

    prs_dataset.set_backend("numpy")

    data_standardized = prs_dataset.scaled_data

    if data_standardized:
        prs_dataset.inverse_standardize_data()

    df_col = [c for c in prs_dataset.data.columns if c.upper().startswith('PC')] + [group_col]

    pc_df = pd.DataFrame(prs_dataset.get_data_columns(df_col),
                         columns=df_col)

    mean_pcs = pc_df.groupby(group_col).mean()

    # Get the reference cluster (if name is not specified):
    if reference_group == "largest":
        uniq_clust, counts = np.unique(pc_df[group_col], return_counts=True)
        reference_group = uniq_clust[np.argmax(counts)]

    if data_standardized:
        prs_dataset.standardize_data()

    return sorted(mean_pcs.index,
                  key=lambda x: np.sqrt(((mean_pcs.loc[x, :] -
                                          mean_pcs.loc[reference_group, :])**2).sum()))


def rank_individuals_by_pc_distance(prs_dataset,
                                    reference='median',
                                    n_clusters=5):

    assert reference in ['median', 'mean'], f"Reference must be either 'median' or 'mean'. Got: {reference}."

    prs_dataset.set_backend("numpy")

    data_standardized = prs_dataset.scaled_data

    if data_standardized:
        prs_dataset.inverse_standardize_data()

    pc_cols = [c for c in prs_dataset.data.columns if c.upper().startswith('PC')]

    pc_df = pd.DataFrame(prs_dataset.get_data_columns(pc_cols),
                         columns=pc_cols)

    if reference == 'median':
        ref_val = pc_df.median(axis=0)
    elif reference == 'mean':
        ref_val = pc_df.mean(axis=0)

    if data_standardized:
        prs_dataset.standardize_data()

    return pd.qcut(np.sqrt(((pc_df - ref_val)**2).sum(axis=1)), n_clusters,
                   labels=[f"PC_DIST (Q{i})" for i in range(1, n_clusters+1)])


