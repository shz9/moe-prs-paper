import copy
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class PRSDataset(Dataset):
    """
    A PyTorch Dataset class for handling PRS and phenotype data.
    """

    def __init__(
        self,
        analysis_id,
        dataframe,
        phenotype_col,
        meta_cols=None,
        prs_cols=None,
        covariates_cols=None,
        group_getitem_cols=None,
        phenotype_likelihood="infer",
        backend="numpy",
    ):
        """
        :param dataframe: A pandas DataFrame containing the PRS, covariates, and phenotype data.
        :param phenotype_col: The name of the column containing the phenotype data.
        :param meta_cols: A list of column names for the metadata (e.g. individual IDs / attributes that we don't
        necessarily need to use in the regression models).
        :param prs_cols: A list of column names for the PRS models.
        :param covariates_cols: A list of column names for the covariates.
        :param group_getitem_cols: A dictionary mapping categories of data to the relevant keys from the
         pandas dataframe. This is useful for iterative data fetching (e.g. data loaders).
            These are used to define what columns/groups of columns are fetched in the __getitem__ method.
        :param phenotype_likelihood: The likelihood of the phenotype data. If 'infer', the likelihood is inferred
        from the data (i.e. binomial if the phenotype is binary, Gaussian otherwise).
        :param backend: The backend for the data (i.e. torch Tensor or numpy arrays).

        """

        assert backend in ("torch", "numpy")
        assert phenotype_likelihood in ("infer", "gaussian", "binomial")

        if group_getitem_cols is not None and dataframe is not None:
            for cols in group_getitem_cols.values():
                assert all([c in dataframe.columns for c in cols])

        self._analysis_id = analysis_id

        self.backend = None
        self.set_backend(backend)

        self.data = dataframe

        # Extract the column names for the metadata (e.g. individual IDs / attributes that we don't
        # necessarily need to use in the regression models):
        self.meta_cols = meta_cols
        if self.meta_cols is not None and isinstance(self.meta_cols, str):
            self.meta_cols = [self.meta_cols]

            # Check that all the provided columns are present in the dataframe:
            assert all([c in self.data.columns for c in self.meta_cols])

        # Extract the column name for the phenotype:
        self.phenotype_col = phenotype_col
        # Check that the provided column is present in the dataframe:
        assert self.phenotype_col in self.data.columns

        # Extract the column names for the PRS models:
        self.prs_cols = prs_cols
        if self.prs_cols is not None and isinstance(self.prs_cols, str):
            self.prs_cols = [self.prs_cols]

            # Check that all the provided columns are present in the dataframe:
            assert all([c in self.data.columns for c in self.prs_cols])

        # For consistency, we sort the PRS IDs:
        if self.prs_cols is not None:
            self.prs_cols = sorted(self.prs_cols)

        # Extract the column names for the covariates:
        self.covariates_cols = covariates_cols
        if self.covariates_cols is not None and isinstance(self.covariates_cols, str):
            self.covariates_cols = [self.covariates_cols]

            # Check that all the provided columns are present in the dataframe:
            assert all([c in self.data.columns for c in self.covariates_cols])

        # For consistency, we sort the covariate IDs:
        if self.covariates_cols is not None:
            self.covariates_cols = sorted(self.covariates_cols)

        if phenotype_likelihood == "infer":
            if len(np.unique(self.data[phenotype_col].values)) == 2:
                self.phenotype_likelihood = "binomial"
            else:
                self.phenotype_likelihood = "gaussian"
        else:
            self.phenotype_likelihood = phenotype_likelihood

        # Extract and keep the relevant data columns:

        # Re-order the dataset columns so that the metadata columns come first:
        self.data = self.data[self.get_data_cols(exclude_meta=False)]

        # Define the getitem columns:
        self.group_getitem_cols = None
        self.group_getitem_col_idx = None
        self.set_group_getitem_cols(group_getitem_cols)

        # Utilities for transforming the data:
        self.scaled_data = False
        self.scaler = StandardScaler()

        self._data_matrix = None  # cached float32 matrix
        self._cache_cols = None
        self._cache_group_getitem_col_idx = None

    @classmethod
    def from_pickle(cls, f, backend="numpy"):
        """
        Load a PRSDataset object from a pickle file.
        """
        with open(f, "rb") as dat_f:
            obj = pickle.load(dat_f)

        if backend is not None:
            obj.set_backend(backend)

        return obj

    @property
    def n_prs_models(self):
        """
        Get the number of PRS models in the dataset.
        """
        if self.prs_cols is not None:
            return len(self.prs_cols)

    @property
    def n_covariates(self):
        """
        Get the number of covariates in the dataset.
        """
        if self.covariates_cols is not None:
            return len(self.covariates_cols)

    @property
    def N(self):
        """
        Get the number of samples in the dataset.
        """
        return self.data.shape[0]

    @property
    def prs_ids(self):
        """
        Get the names/IDs of the experts (i.e. PRS models).
        """
        return self.prs_cols

    @property
    def covariates(self):
        """
        Get the covariates (sample attributes) from the dataset.
        """
        return self.covariates_cols

    @property
    def analysis_id(self):
        return self._analysis_id

    def cache_data_matrix(self):
        """
        Cache ONLY the columns needed by group_getitem_cols as float32.
        """

        assert self.group_getitem_cols is not None, "Call set_group_getitem_cols first."

        # collect used columns in dataframe order
        used = []
        used_set = set()
        for cols in self.group_getitem_cols.values():
            for c in cols:
                if c not in used_set:
                    used.append(c)
                    used_set.add(c)

        # sanity: all used columns must be numeric (for now atleast)
        bad = [c for c in used if not pd.api.types.is_numeric_dtype(self.data[c])]
        if bad:
            raise ValueError(
                f"Non-numeric columns are used by group_getitem_cols: {bad}"
            )

        self._cache_cols = used
        X = self.data[self._cache_cols].to_numpy(dtype=np.float32, copy=True)

        # build cached idx mapping
        col_to_pos = {c: j for j, c in enumerate(self._cache_cols)}
        self._cache_group_getitem_col_idx = {
            k: [col_to_pos[c] for c in cols]
            for k, cols in self.group_getitem_cols.items()
        }

        if self.backend is torch.Tensor:
            self._data_matrix = torch.from_numpy(X)
        else:
            self._data_matrix = X

    def get_batch(self, indices):
        """
        Vectorized batch fetch: indices are original row indices into self.data.
        Returns the same dict structure as __getitem__ but batched.
        """
        assert self._data_matrix is not None, "Call cache_data_matrix() first."
        assert self._cache_group_getitem_col_idx is not None, "Cache mapping missing."

        if isinstance(indices, list):
            indices = np.asarray(indices)

        if self.backend is torch.Tensor:
            if isinstance(indices, np.ndarray):
                indices = torch.as_tensor(indices, dtype=torch.long)
            Xb = self._data_matrix.index_select(0, indices)
            return {k: Xb[:, v] for k, v in self._cache_group_getitem_col_idx.items()}
        else:
            Xb = self._data_matrix[indices]
            return {k: Xb[:, v] for k, v in self._cache_group_getitem_col_idx.items()}

    def get_data_cols(self, exclude_meta=True):
        # Extract and keep the relevant data columns:
        data_cols = []

        if self.meta_cols is not None and not exclude_meta:
            data_cols.extend(self.meta_cols)
        if self.prs_cols is not None:
            data_cols.extend(self.prs_cols)
        if self.covariates_cols is not None:
            data_cols.extend(self.covariates_cols)

        data_cols += [self.phenotype_col]

        return data_cols

    @property
    def continuous_cols(self):
        """
        Get a list of columns in data that have continuous values
        """

        return np.array(
            [
                c
                for c in self.get_data_cols(exclude_meta=True)
                if len(np.unique(self.data[c].values)) > 2
            ]
        )

    @property
    def binary_cols(self):
        """
        Get a list of columns in data that have binary values
        """
        np.array(
            [
                c
                for c in self.get_data_cols(exclude_meta=True)
                if c not in self.continuous_cols
            ]
        )

    def set_backend(self, new_backend):
        """
        Set the backend for the data (i.e. Torch Tensor or numpy arrays).
        """

        if new_backend == "numpy":
            self.backend = np.array
        elif new_backend == "torch":
            self.backend = torch.Tensor
        else:
            raise NotImplementedError(f"Backend {new_backend} not recognized!")

    def standardize_data(self, scaler=None, refit=False):
        """
        Standardize the data so that each column has a mean of 0 and a standard deviation of 1.
        :param scaler: A StandardScaler object to use for standardization.
        :param refit: If True, refit the scaler even if it has already been fit.
        """

        std_cols = (
            list(self.covariates_cols) + [self.phenotype_col] + list(self.prs_cols)
        )
        std_cols = [c for c in std_cols if c in self.continuous_cols]

        if scaler is None:
            assert self.scaler is not None

            if refit or not hasattr(self.scaler, "n_features_in_"):
                self.scaler.fit(self.data[std_cols])
            else:
                return
        else:
            # Sanity checks:
            if hasattr(scaler, "feature_names_in_"):
                if not np.array_equal(scaler.feature_names_in_, std_cols):
                    # Ensure that all the features defined in the scaler do exist in the dataframe:
                    assert all([c in std_cols for c in scaler.feature_names_in_])

                    # If they do, then use the scaler features as the columns to standardize:
                    std_cols = scaler.feature_names_in_

            elif hasattr(scaler, "n_features_in_"):
                assert scaler.n_features_in_ == len(std_cols)

            # Check if the data is already scaled (if it is, then we need to reverse-transform it first):
            if self.scaled_data:
                self.inverse_standardize_data()

            self.scaler = scaler

        self.data[std_cols] = self.scaler.transform(self.data[std_cols])
        self.scaled_data = True

    def inverse_standardize_data(self):
        """
        Reverse-transform the data so it returns to its original scale.
        """

        assert self.scaler is not None

        if self.scaled_data:
            if self.scaler.feature_names_in_ is not None:
                std_cols = self.scaler.feature_names_in_
            else:
                std_cols = (
                    list(self.covariates_cols)
                    + [self.phenotype_col]
                    + list(self.prs_cols)
                )
                std_cols = [c for c in std_cols if c in self.continuous_cols]

            self.data[std_cols] = self.scaler.inverse_transform(self.data[std_cols])
            self.scaled_data = False

    def adjust_phenotype_for_covariates(self):
        """
        Adjust the phenotype for the covariates.
        """

        assert self.covariates_cols is not None
        assert self.phenotype_likelihood == "gaussian"

        from magenpy.stats.transforms.phenotype import adjust_for_covariates

        self.data[self.phenotype_col] = adjust_for_covariates(
            self.data[self.phenotype_col], self.get_covariates()
        )

    def adjust_prs_for_covariates(self):
        """
        Adjust the PRS for the covariates.
        """

        assert self.covariates_cols is not None

        from magenpy.stats.transforms.phenotype import adjust_for_covariates

        for prs_col in self.prs_cols:
            self.data[prs_col] = adjust_for_covariates(
                self.data[prs_col], self.get_covariates()
            )

    def filter_outliers(self):
        """
        Filter outliers in both phenotype and PRS columns.
        """

        assert self.phenotype_col is not None
        assert self.prs_cols is not None

        from magenpy.stats.transforms.phenotype import detect_outliers

        outliers = detect_outliers(self.data[self.phenotype_col])
        for prs_col in self.prs_cols:
            outliers |= detect_outliers(self.data[prs_col])

        # Filter out the outliers:
        self.filter_samples(np.where(~outliers)[0])

    def filter_samples(self, keep_idx):
        if keep_idx.dtype == bool:
            self.data = self.data.loc[keep_idx, :]
        else:
            self.data = self.data.iloc[keep_idx, :]

    def drop_prs(self, prs_ids):
        """
        Drop certain polygenic scores from the dataset
        """

        if isinstance(prs_ids, str):
            prs_ids = [prs_ids]

        for pid in prs_ids:
            if pid not in self.prs_cols:
                raise KeyError(f"The PRS ID {pid} is not present in this dataset!")

        self.prs_cols = [pid for pid in self.prs_cols if pid not in prs_ids]
        self.data.drop(columns=prs_ids, inplace=True)

    def train_test_split(self, test_size, inplace=False):
        assert 0 < test_size < self.N

        from sklearn.model_selection import train_test_split

        if self.phenotype_likelihood == "binomial":
            stratify = self.get_phenotype()
        else:
            stratify = None

        train_idx, test_idx = train_test_split(
            np.arange(self.N), test_size=test_size, stratify=stratify
        )

        test_dataset = copy.deepcopy(self)
        test_dataset.filter_samples(test_idx)

        if inplace:
            train_dataset = self
        else:
            train_dataset = copy.deepcopy(self)

        train_dataset.filter_samples(train_idx)

        return train_dataset, test_dataset

    def get_prs_predictions(self, scaler=None):
        """
        Get the PRS predictions from the dataset.
        :param scaler: A StandardScaler object to use for standardization. Must match the ordering/dimensions of
        the entire dataset (no explicit checks for this at the moment).
        """
        return self.get_data_columns(self.prs_ids, scaler=scaler)

    def get_covariates(self, scaler=None):
        """
        Get the covariates from the dataset.
        :param scaler: A StandardScaler object to use for standardization. Must match the ordering/dimensions of
        the entire dataset (no explicit checks for this at the moment).
        """
        if self.covariates_cols is not None:
            return self.get_data_columns(self.covariates_cols, scaler=scaler)

    def get_phenotype(self, scaler=None):
        """
        Get the phenotype column from the dataset.
        :param scaler: A StandardScaler object to use for standardization. Must match the ordering/dimensions of
        the entire dataset (no explicit checks for this at the moment).
        """

        return self.get_data_columns(self.phenotype_col, scaler=scaler)

    def get_data_columns(self, columns, add_intercept=False, scaler=None):
        """
        Get data for a subset of columns.
        :param columns: A list of column names to extract from the dataset.
        :param add_intercept: If True, add a column of ones to the left of the data matrix.
        :param scaler: A StandardScaler object to use for standardization. Must match the ordering/dimensions of
        the entire dataset (no explicit checks for this at the moment).
        """

        if isinstance(columns, str):
            columns = [columns]

        # Check that the requested columns are valid and present in the dataset:
        assert all([c in self.data.columns for c in columns])

        # --------------------------------------------------------------
        # If a new scaler is provided, we need to make sure to transform
        # the data according to this scaler before fetching the data:

        curr_scaled = self.scaled_data  # Is the data currently scaled?

        # If a scaler is provided, use it to transform the data:
        if scaler is not None:
            self.standardize_data(scaler=scaler)

        # Fetch the data:
        data = self.data[columns].values

        # If a scaler was provided, reverse-transform the data back to its original scale:
        if scaler is not None:
            self.inverse_standardize_data()
            if curr_scaled:
                self.standardize_data(refit=True)
            else:
                self.scaler = StandardScaler()  # Reset the scaler to an empty one

        # --------------------------------------------------------------

        if add_intercept:
            data = np.hstack([np.ones((data.shape[0], 1)), data])

        return self.backend(data)

    def get_ancestry(self, ancestry_col: str = "Ancestry"):
        """
        Return the ancestry labels as a 1D numpy array.

        By default, assumes the column is named 'Ancestry'.
        Change ancestry_col if your column has a different name.
        """
        assert ancestry_col in self.data.columns, (
            f"Ancestry column '{ancestry_col}' not found in dataset columns: "
            f"{list(self.data.columns)}"
        )
        return self.data[ancestry_col].values

    def concatenate(self, prs_dataset):
        """
        Concatenate the current PRSDataset (self) to another dataset.
        In principle, this should be the same as `pd.concat`,
        but we need to make sure same columns are present and that
        both datasets are not scaled before merging.

        By default, this returns a new PRSDataset object.

        """

        # Check that the columns are the same:
        assert all([c in self.data.columns for c in prs_dataset.data.columns])
        assert self.phenotype_col == prs_dataset.phenotype_col
        assert self.meta_cols == prs_dataset.meta_cols
        assert self.prs_cols == prs_dataset.prs_cols
        assert self.covariates_cols == prs_dataset.covariates_cols

        if self.scaled_data:
            self.inverse_standardize_data()

        if prs_dataset.scaled_data:
            prs_dataset.inverse_standardize_data()

        return PRSDataset(
            pd.concat([self.data, prs_dataset.data], axis=0),
            phenotype_col=self.phenotype_col,
            meta_cols=self.meta_cols,
            prs_cols=self.prs_cols,
            covariates_cols=self.covariates_cols,
            phenotype_likelihood=self.phenotype_likelihood,
        )

    def set_group_getitem_cols(self, group_getitem_cols):
        """
        Set the group_getitem_cols attribute.
        """
        self.group_getitem_cols = group_getitem_cols
        if self.group_getitem_cols is not None:
            self.group_getitem_col_idx = {
                k: [list(self.data.columns).index(c) for c in v]
                for k, v in group_getitem_cols.items()
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert self.group_getitem_col_idx is not None

        # ---- Cached fast path (indexes into cached matrix) ----
        if self._data_matrix is not None:
            assert self._cache_group_getitem_col_idx is not None, (
                "Call cache_data_matrix() first."
            )
            row = self._data_matrix[idx]  # row is over _cache_cols
            return {k: row[v] for k, v in self._cache_group_getitem_col_idx.items()}

        # ---- Non-cached path (indexes into full dataframe row) ----
        row = self.data.iloc[
            idx, :
        ].values  # this can be dtype=object if there are strings elsewhere
        out = {}
        for k, v in self.group_getitem_col_idx.items():
            # IMPORTANT: only cast the selected slice (v) to float32, NOT the whole row
            out[k] = self.backend(row[v].astype(np.float32, copy=False))
        return out

    def save(self, f):
        with open(f, "wb") as opf:
            pickle.dump(self, opf)
