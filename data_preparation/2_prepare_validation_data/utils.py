import numpy as np


def detect_outliers(phenotype, sigma_threshold=3, stratify=None, nan_policy='omit'):
    """
    Detect samples with outlier phenotype values.
    This function takes a vector of quantitative phenotypes,
    computes the z-score for every individual, and returns a
    boolean vector indicating whether individual i has phenotype value
    within the specified standard deviations `sigma_threshold`.
    :param phenotype: A numpy vector of continuous or quantitative phenotypes.
    :param sigma_threshold: The multiple of standard deviations or sigmas after
    which we consider the phenotypic value an outlier.
    :param stratify: A numpy array indicating group membership to stratify the outlier detection.
    :param nan_policy: The policy to use when encountering NaN values in the phenotype vector.
    By default, we compute the z-scores ignoring NaN values.

    :return: A boolean array indicating whether the phenotype value is an outlier (i.e.
    True indicates outlier).
    """

    from scipy.stats import zscore

    if stratify is None:
        stratify = np.ones_like(phenotype)

    mask = np.zeros_like(phenotype, dtype=bool)

    for group in np.unique(stratify):
        mask[stratify == group] = np.abs(zscore(phenotype[stratify == group],
                                                nan_policy=nan_policy)) > sigma_threshold

    return mask
