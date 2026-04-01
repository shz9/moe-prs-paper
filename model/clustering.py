from sklearn.cluster import HDBSCAN


def cluster_data(
    dataset, data_columns=None, clustering_module=None, standardize=True, n_jobs=None
):
    if clustering_module is None:
        clustering_module = HDBSCAN(min_cluster_size=50, n_jobs=n_jobs)

    if data_columns is None:
        # Cluster based on the covariates:
        input = dataset.get_data_columns(dataset.covariates + [dataset.phenotype_col])
    else:
        # Cluster based on columns requested by user:
        input = dataset.get_data_columns(data_columns)

    if standardize:
        input = (input - input.mean(axis=0)) / input.std(axis=0)

    return clustering_module.fit(dataset.get_covariates()).labels_
