from sklearn.cluster import HDBSCAN


def cluster_covariates(dataset, clustering_module=None):
    if clustering_module is None:
        clustering_module = HDBSCAN(min_cluster_size=50)

    return clustering_module.fit(dataset.get_covariates()).labels_
