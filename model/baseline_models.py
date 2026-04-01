import copy
import pickle

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.utils.validation import check_is_fitted


class MultiPRS(object):
    def __init__(
        self,
        prs_dataset=None,
        expert_cols=None,
        covariates_cols=None,
        class_weights=None,
        add_intercept=True,
        standardize_data=True,
        penalty_type=None,
        penalty=0.0,
    ):
        # -------------------------------------------------------------------------
        # Sanity checks:

        assert penalty >= 0.0
        if penalty > 0.0:
            assert penalty_type is not None

        # -------------------------------------------------------------------------
        # Process / extract training data:

        # Initialize the quantities used to hold the data:
        self.phenotype = None
        self.input_data = None

        # Initialize the data scaler:
        self.data_scaler = None

        # Initialize / store the names of the columns to be used as inputs:
        self.covariates_cols = covariates_cols
        if self.covariates_cols is not None and isinstance(self.covariates_cols, str):
            self.covariates_cols = [self.covariates_cols]
        self.expert_cols = expert_cols
        if self.expert_cols is not None and isinstance(self.expert_cols, str):
            self.expert_cols = [self.expert_cols]

        if prs_dataset is not None:
            # If standardize_data is True, standardize the training data:
            if standardize_data:
                prs_dataset.standardize_data()
                self.data_scaler = copy.deepcopy(prs_dataset.scaler)

            # Process the phenotype data:
            self.phenotype = prs_dataset.get_phenotype().reshape(-1, 1)

            # Process the input data:
            if self.input_cols is not None:
                self.input_data = prs_dataset.get_data_columns(self.input_cols)

        # -------------------------------------------------------------------------

        # Determine the family of the phenotype:
        if prs_dataset is not None:
            self.family = prs_dataset.phenotype_likelihood
        else:
            self.family = None

        # Initialize the regression model:
        if self.family == "gaussian":
            if penalty_type == "l1":
                self.reg_model = Lasso(alpha=penalty, fit_intercept=add_intercept)
            elif penalty_type == "l2":
                self.reg_model = Ridge(alpha=penalty, fit_intercept=add_intercept)
            elif penalty_type == "elasticnet":
                self.reg_model = ElasticNet(alpha=penalty, fit_intercept=add_intercept)
            else:
                self.reg_model = LinearRegression(fit_intercept=add_intercept)
        else:
            penalty_params = {"penalty": penalty_type}

            if penalty_type is not None:
                if penalty == 0:
                    penalty_params["C"] = np.inf
                else:
                    penalty_params["C"] = penalty

            self.reg_model = LogisticRegression(
                fit_intercept=add_intercept,
                class_weight=class_weights,
                **penalty_params,
            )

    @classmethod
    def from_saved_model(cls, param_file):
        model = cls()

        with open(param_file, "rb") as pf:
            (
                model.reg_model,
                model.expert_cols,
                model.covariates_cols,
                model.data_scaler,
                model.family,
            ) = pickle.load(pf)

        return model

    @property
    def N(self):
        """
        The number of samples
        """
        if self.phenotype is not None:
            return self.phenotype.shape[0]

    @property
    def K(self):
        """
        The number of experts
        """
        if self.expert_cols is not None:
            return len(self.expert_cols)

    @property
    def C(self):
        """
        The number of input covariates to use when tuning the PRS
        """
        if self.covariates_cols is not None:
            return len(self.covariates_cols)

    @property
    def input_cols(self):
        """
        The names of the input columns
        """
        input_cols = []
        if self.covariates_cols is not None:
            input_cols += list(self.covariates_cols)
        if self.expert_cols is not None:
            input_cols += list(self.expert_cols)

        if len(input_cols) > 0:
            return input_cols

    def predict(self, prs_dataset=None, logit_scale=False):
        """
        Predict for training or new test samples.

        :param prs_dataset: An independent test set.
        :param logit_scale: If we're analyzing binary phenotypes
        the user can request returning results on logit scale.
        """
        assert self.input_cols is not None

        if prs_dataset is None:
            input_data = self.input_data
        else:
            input_data = prs_dataset.get_data_columns(
                self.input_cols, scaler=self.data_scaler
            )

        if self.family == "gaussian":
            return self.reg_model.predict(input_data).flatten()
        else:
            proba = self.reg_model.predict_proba(input_data)[:, 1]

            if logit_scale:
                return logit(proba)
            else:
                return proba

    def predict_prs(self, prs_dataset=None, logit_scale=False):
        assert self.input_cols is not None

        coefs = self.get_coefficients()

        if prs_dataset is None:
            input_data = self.input_data[:, [0, self.C]["Covariates" in coefs] :]
        else:
            input_data = prs_dataset.get_data_columns(
                self.expert_cols, scaler=self.data_scaler
            )

        pred = input_data.dot(coefs["PRS"]).flatten()

        if self.family == "gaussian":
            return pred
        else:
            if logit_scale:
                return pred
            else:
                return expit(pred)

    def get_coefficients(self):
        assert self.reg_model is not None

        coefs = {"Intercept": self.reg_model.intercept_}

        if self.expert_cols is not None:
            if self.covariates_cols is not None:
                coefs["Covariates"] = pd.DataFrame(
                    self.reg_model.coef_.flatten()[: self.C],
                    index=self.covariates_cols,
                    columns=["Coefficient"],
                )

                coefs["PRS"] = pd.DataFrame(
                    self.reg_model.coef_.flatten()[self.C :],
                    index=self.expert_cols,
                    columns=["Coefficient"],
                )
            else:
                coefs["PRS"] = pd.DataFrame(
                    self.reg_model.coef_.T,
                    index=self.expert_cols,
                    columns=["Coefficient"],
                )
        else:
            coefs["Covariates"] = pd.DataFrame(
                self.reg_model.coef_.T,
                index=self.covariates_cols,
                columns=["Coefficient"],
            )

        return coefs

    def fit(self):
        """
        Fit the model to the training data
        """
        assert self.input_data is not None
        assert self.phenotype is not None

        self.reg_model = self.reg_model.fit(self.input_data, self.phenotype)

        return self

    def save(self, output_file):
        """
        Save the parameters of the model to file.
        """

        try:
            check_is_fitted(self.reg_model)
        except NotFittedError:
            raise NotFittedError(
                "The model has not been fitted yet. Call `.fit() first."
            )

        with open(output_file, "wb") as outf:
            pickle.dump(
                [
                    self.reg_model,
                    self.expert_cols,
                    self.covariates_cols,
                    self.data_scaler,
                    self.family,
                ],
                outf,
            )


class AncestryWeightedPRS(object):
    """
    Model PRS using pre-defined ancestry weights (e.g. from a classifier).
    Supports two weighing schemes:
      - "before": combine ancestry-weighted expert PRSs into a single PRS column,
                  then fit one model on (covariates + combined PRS).
      - "after":  fit one model per ancestry (using the expert PRS corresponding
                  to that ancestry) and combine ancestry-specific predictions
                  by weighting them with the ancestry weights.

    Behavior added: samples that do not have **more than** `min_rep_prop`
    of their ancestry represented by the provided polygenic scores are:
      - excluded from training (omitted),
      - return NaN at prediction time.

    Parameters
    ----------
    prs_dataset : object or None
        Dataset object exposing:
          - standardize_data(), scaler attribute after standardize_data(),
          - get_phenotype() -> ndarray (N,) or (N,1),
          - get_data_columns(colnames, scaler=None) -> ndarray (N, ncols),
          - phenotype_likelihood attribute ("gaussian" or else treated as binary).
        If None, the class is created empty and can be populated via from_saved_model.
    expert_cols : list or str or None
        Names of the PRS columns (experts). If string, converted to list.
    covariates_cols : list or str or None
        Names of covariate columns to include (may be None).
    expert_ancestry_map : dict or None
        Mapping {expert_col_name: ancestry_weight_col_name}. Required if prs_dataset
        and expert_cols are provided.
    weighing_scheme : {"before", "after"}
        Whether to combine PRS columns before fitting ("before") or fit per-ancestry
        and combine predictions afterwards ("after").
    class_weights : dict or "balanced" or None
        Passed to LogisticRegression as class_weight for binary phenotypes.
    add_intercept : bool
        Whether to fit an intercept in the regression models.
    standardize_data : bool
        If True, call prs_dataset.standardize_data() and keep a copy of the scaler.
    penalty_type : {"l1","l2","elasticnet",None}
        Penalty type used for regularized models (for logistic/regression).
    penalty : float
        Penalty strength. If > 0, penalty_type must be provided. For logistic, this
        becomes C (inverse reg) with sklearn semantics; if penalty==0.0 we set C=inf.
    min_rep_prop : float, default 0.5
        Minimum fraction (p) of a sample's ancestry that must be represented by the
        provided PRSs. A sample is considered represented if:
            sum(ancestry_weights over provided ancestry columns) > min_rep_prop
        NOTE: the comparison is strict (">") to match the "more than p" requirement.
    """

    def __init__(
        self,
        prs_dataset=None,
        expert_cols=None,
        covariates_cols=None,
        expert_ancestry_map=None,
        weighing_scheme="after",
        class_weights=None,
        add_intercept=True,
        standardize_data=True,
        penalty_type=None,
        penalty=0.0,
        min_rep_prop=0.5,
    ):
        # -------------------------------------------------------------------------
        # Sanity checks:
        assert 0.0 <= min_rep_prop < 1.0
        assert penalty >= 0.0
        if penalty > 0.0:
            assert penalty_type is not None

        assert weighing_scheme in ("before", "after")
        if prs_dataset is not None and expert_cols is not None:
            assert expert_ancestry_map is not None

        # -------------------------------------------------------------------------
        # Store config & initialize internals:
        self.min_rep_prop = min_rep_prop
        self.weighing_scheme = weighing_scheme
        self.class_weights = class_weights

        # covariates + experts normalization to lists
        self.covariates_cols = covariates_cols
        if self.covariates_cols is not None and isinstance(self.covariates_cols, str):
            self.covariates_cols = [self.covariates_cols]

        self.expert_cols = expert_cols
        if self.expert_cols is not None and isinstance(self.expert_cols, str):
            self.expert_cols = [self.expert_cols]

        # expert -> ancestry weight mapping
        self.expert_ancestry_map = expert_ancestry_map
        self.ancestry_weights_cols = None

        if self.expert_ancestry_map is not None:
            # keep only expert cols present in the map
            self.expert_cols = [
                col for col in self.expert_cols if col in self.expert_ancestry_map
            ]
            # ancestry weights corresponding to the kept experts (order matches expert_cols)
            self.ancestry_weights_cols = [
                self.expert_ancestry_map[col] for col in self.expert_cols
            ]
            # filter expert_ancestry_map accordingly
            self.expert_ancestry_map = {
                k: v
                for k, v in self.expert_ancestry_map.items()
                if k in self.expert_cols
            }

            assert len(self.expert_cols) > 1
            assert len(self.ancestry_weights_cols) > 1

        # data holders
        self.phenotype = None
        self.input_data = None  # structured depending on weighing_scheme
        self.data_scaler = None
        self.family = None
        self.reg_model = None

        # model hyperparams (kept to construct models)
        self._penalty_type = penalty_type
        self._penalty = penalty
        self._add_intercept = add_intercept

        # If dataset provided, prepare data and instantiate models
        if prs_dataset is not None:
            if standardize_data:
                prs_dataset.standardize_data()
                self.data_scaler = copy.deepcopy(prs_dataset.scaler)

            self.phenotype = prs_dataset.get_phenotype().reshape(-1, 1)
            self.input_data = self._extract_input_data(prs_dataset)
            self.family = prs_dataset.phenotype_likelihood

        # instantiate models (single or per-ancestry)
        if self.input_data is not None:
            # build base estimator
            if self.family == "gaussian":
                if self._penalty_type == "l1":
                    base = Lasso(alpha=self._penalty, fit_intercept=add_intercept)
                elif self._penalty_type == "l2":
                    base = Ridge(alpha=self._penalty, fit_intercept=add_intercept)
                elif self._penalty_type == "elasticnet":
                    base = ElasticNet(alpha=self._penalty, fit_intercept=add_intercept)
                else:
                    base = LinearRegression(fit_intercept=add_intercept)
            else:
                C = self._penalty if (self._penalty and self._penalty > 0.0) else np.inf
                base = LogisticRegression(
                    fit_intercept=add_intercept,
                    class_weight=self.class_weights,
                    penalty=self._penalty_type,
                    C=C,
                    max_iter=1000,
                )

            if self.weighing_scheme == "before":
                self.reg_model = base
            else:
                # one model per ancestry weight column present in input_data
                self.reg_model = {
                    k: copy.deepcopy(base) for k in self.input_data.keys()
                }

    @classmethod
    def from_saved_model(cls, param_file):
        model = cls()
        with open(param_file, "rb") as pf:
            (
                model.reg_model,
                model.expert_cols,
                model.expert_ancestry_map,
                model.ancestry_weights_cols,
                model.covariates_cols,
                model.weighing_scheme,
                model.data_scaler,
                model.family,
                model.min_rep_prop,
            ) = pickle.load(pf)
        return model

    @property
    def N(self):
        if self.phenotype is not None:
            return self.phenotype.shape[0]

    @property
    def K(self):
        if self.expert_cols is not None:
            return len(self.expert_cols)

    @property
    def C(self):
        if self.covariates_cols is not None:
            return len(self.covariates_cols)

    def _extract_input_data(self, prs_dataset, scaler=None):
        """
        Extract relevant input arrays.

        - ancestry_weights: matrix (N x M) from ancestry weight columns corresponding
          to the provided expert PRSs (order matches self.ancestry_weights_cols).
        - prs_data: matrix (N x K) of expert PRSs (order matches self.expert_cols).
        - covariates_data: (N x C) or None.
        Returns:
          If weighing_scheme == "before":
            {"data": ndarray (N x (C+1)), "keep_samples": boolean mask (N,), "weights": None}
          Else (after):
            dict keyed by ancestry weight column name:
              {anc_col: {"data": ndarray (N x (C+1_prs_for_that_anc?)),
                         "keep_samples": boolean mask (N,),
                         "weights": ndarray (N,) } }
        Samples are considered "represented" iff ancestry_weights.sum(axis=1) > min_rep_prop.
        Those failing this check are excluded from training and will get NaN at prediction.
        """
        ancestry_weights = prs_dataset.get_data_columns(self.ancestry_weights_cols)
        prs_data = prs_dataset.get_data_columns(self.expert_cols, scaler=scaler)

        if self.covariates_cols is not None:
            covariates_data = [
                prs_dataset.get_data_columns(self.covariates_cols, scaler=scaler)
            ]
        else:
            covariates_data = []

        # Sum the ancestry weights:
        ancestry_weights_sum = ancestry_weights.sum(axis=1)

        # Strict "more than p" filter:
        general_keep = ancestry_weights_sum > self.min_rep_prop

        if self.weighing_scheme == "before":
            # Construct a weights PRS score:
            # NOTE: Here we normalized the weights per individual so they sum to 1.
            weighted_prs = (
                (prs_data * (ancestry_weights / ancestry_weights_sum.reshape(-1, 1)))
                .sum(axis=1)
                .reshape(-1, 1)
            )
            return {
                "data": np.concatenate(covariates_data + [weighted_prs], axis=1),
                "keep_samples": general_keep,
                "weights": None,
            }
        else:
            input_data = {}
            for idx, anc_col in enumerate(self.ancestry_weights_cols):
                keep_samples = (ancestry_weights[:, idx] > 0) & general_keep
                if keep_samples.sum() > 0:
                    # for 'after', use the corresponding expert PRS as the PRS input
                    prs_for_anc = prs_data[:, idx].reshape(-1, 1)
                    data = np.concatenate(covariates_data + [prs_for_anc], axis=1)
                    input_data[anc_col] = {
                        "data": data,
                        "keep_samples": keep_samples,
                        "weights": ancestry_weights[:, idx],
                    }
            return input_data

    def predict(self, prs_dataset=None):
        """
        Predict. If prs_dataset provided, extract data with stored scaler.
        Samples failing the represented-ancestry threshold receive NaN.
        For binary outcomes, returns probabilities.
        For the "after" scheme, combines ancestry-specific probabilities
        as a normalised linear mixture: P(y=1) = sum_j w_j P(y=1 | ancestry=j).
        """
        if prs_dataset is None:
            input_data = self.input_data
        else:
            input_data = self._extract_input_data(prs_dataset, scaler=self.data_scaler)

        if self.weighing_scheme == "before":
            if self.family == "gaussian":
                pred = self.reg_model.predict(input_data["data"]).flatten()
            else:
                pred = self.reg_model.predict_proba(input_data["data"])[:, 1]
            pred[~input_data["keep_samples"]] = np.nan
        else:
            N = next(iter(input_data.values()))["data"].shape[0]
            pred = np.zeros(N)
            weight_sum = np.zeros(N)
            keep_samples = np.zeros(N, dtype=bool)

            for anc, dat in input_data.items():
                model = self.reg_model.get(anc)
                if model is None:
                    continue
                X = dat["data"]
                if self.family == "gaussian":
                    pred_anc = model.predict(X).flatten()
                else:
                    pred_anc = model.predict_proba(X)[:, 1]
                pred += pred_anc * dat["weights"]
                weight_sum += dat["weights"] * dat["keep_samples"]
                keep_samples |= dat["keep_samples"]

            pred[keep_samples] /= weight_sum[keep_samples]
            pred[~keep_samples] = np.nan

        return pred

    def predict_prs(self, prs_dataset=None, logit_scale=False):
        """
        PRS-only prediction (excludes covariate effects).

        Uses the same data extraction pipeline as predict(), but applies only
        the PRS coefficient from each fitted model, ignoring covariate
        contributions.

        For the "after" scheme with binary outcomes, combines
        ancestry-specific probabilities as a normalised linear mixture,
        consistent with predict(). The logit_scale flag is only meaningful
        for the "before" scheme or Gaussian outcomes.
        """
        if prs_dataset is None:
            input_data = self.input_data
        else:
            input_data = self._extract_input_data(prs_dataset, scaler=self.data_scaler)

        if self.weighing_scheme == "before":
            prs_col = input_data["data"][:, -1]
            coef = np.asarray(self.reg_model.coef_).ravel()
            # intercept = float(np.asarray(self.reg_model.intercept_).ravel()[0])
            # pred = intercept + prs_col * coef[-1]
            # ignoring intercept for now:
            pred = prs_col * coef[-1]
            pred[~input_data["keep_samples"]] = np.nan
            if self.family != "gaussian" and not logit_scale:
                pred = expit(pred)
        else:
            N = next(iter(input_data.values()))["data"].shape[0]
            pred = np.zeros(N)
            weight_sum = np.zeros(N)
            keep_samples = np.zeros(N, dtype=bool)

            for anc, dat in input_data.items():
                mdl = self.reg_model.get(anc)
                if mdl is None:
                    continue
                prs_col = dat["data"][:, -1]
                coef = np.asarray(mdl.coef_).ravel()
                # intercept = float(np.asarray(mdl.intercept_).ravel()[0])
                # lin = intercept + prs_col * coef[-1]
                # IGNORING INTERCEPT FOR NOW:
                lin = prs_col * coef[-1]
                # Convert to probability space before mixing, consistent with predict()
                pred_anc = lin if self.family == "gaussian" else expit(lin)
                pred += pred_anc * dat["weights"]
                weight_sum += dat["weights"] * dat["keep_samples"]
                keep_samples |= dat["keep_samples"]

            pred[keep_samples] /= weight_sum[keep_samples]
            pred[~keep_samples] = np.nan

            # For binary outcomes, pred is already in probability space.
            # logit_scale is not meaningful here since we mixed in probability
            # space — converting back would not recover a proper logit.
            if self.family != "gaussian" and logit_scale:
                raise ValueError(
                    "logit_scale=True is not supported for the 'after' scheme with "
                    "binary outcomes, as predictions are combined in probability space."
                )

        return pred.flatten()

    def get_coefficients(self):
        """
        Return a dict of coefficients:
          - if weighing_scheme == "before": {"All": {"Intercept": ..., "Covariates": ..., "PRS": ...}}
          - else: {anc_col: {...}, ...}
        """
        assert self.reg_model is not None

        if self.weighing_scheme == "before":
            model_map = {"All": self.reg_model}
        else:
            model_map = self.reg_model

        all_coefs = {}
        for k, v in model_map.items():
            coefs = {"Intercept": v.intercept_}
            if self.expert_cols is not None:
                if self.covariates_cols is not None:
                    coefs["Covariates"] = v.coef_.flatten()[: self.C]
                    coefs["PRS"] = v.coef_.flatten()[self.C :]
                else:
                    coefs["PRS"] = v.coef_.flatten()
            else:
                coefs["Covariates"] = v.coef_.flatten()
            all_coefs[k] = coefs
        return all_coefs

    def fit(self):
        """
        Fit model(s). Samples that do not meet the represented-ancestry threshold
        are excluded from training.
        """
        assert self.input_data is not None
        assert self.phenotype is not None

        if self.weighing_scheme == "before":
            keep = self.input_data["keep_samples"]
            self.reg_model = self.reg_model.fit(
                self.input_data["data"][keep], self.phenotype[keep]
            )
        else:
            for anc, dat in self.input_data.items():
                keep = dat["keep_samples"]
                # fit with sample weights for this ancestry
                self.reg_model[anc] = self.reg_model[anc].fit(
                    dat["data"][keep],
                    self.phenotype[keep],
                    sample_weight=dat["weights"][keep],
                )

        return self

    def save(self, output_file):
        """
        Save parameters to disk. Ensures models are fitted before saving.
        """
        try:
            if self.weighing_scheme == "before":
                check_is_fitted(self.reg_model)
            else:
                for _, m in self.reg_model.items():
                    check_is_fitted(m)
        except NotFittedError:
            raise NotFittedError(
                "The model has not been fitted yet. Call `.fit()` first."
            )

        with open(output_file, "wb") as outf:
            pickle.dump(
                [
                    self.reg_model,
                    self.expert_cols,
                    self.expert_ancestry_map,
                    self.ancestry_weights_cols,
                    self.covariates_cols,
                    self.weighing_scheme,
                    self.data_scaler,
                    self.family,
                    self.min_rep_prop,
                ],
                outf,
            )


class AttributePartitionedPRS(object):
    """
    PRS model using a categorical attribute for hard partitioning.

    Each attribute value defines a subgroup:
      - A specific PRS column is used for that subgroup
      - A separate model is fitted per subgroup

    No weighting or mixing is performed.

    Samples whose attribute value is not in attribute_to_score_map:
      - are excluded from training
      - return NaN at prediction time

    Parameters
    ----------
    prs_dataset : object or None
    partition_attribute : str
        Column used to partition individuals (e.g. "Sex")
    attribute_to_score_map : dict
        Mapping {attribute_value: prs_column_name}
    covariates_cols : list or str or None
    class_weights : dict or "balanced" or None
    add_intercept : bool
    standardize_data : bool
    penalty_type : {"l1","l2","elasticnet",None}
    penalty : float
    """

    def __init__(
        self,
        prs_dataset=None,
        partition_attribute=None,
        attribute_to_score_map=None,
        covariates_cols=None,
        class_weights=None,
        add_intercept=True,
        standardize_data=True,
        penalty_type=None,
        penalty=0.0,
    ):
        # ---------------------------
        # Sanity checks
        assert penalty >= 0.0
        if penalty > 0.0:
            assert penalty_type is not None

        if prs_dataset is not None:
            assert partition_attribute is not None
            assert attribute_to_score_map is not None

        self.partition_attribute = partition_attribute
        self.attribute_to_score_map = attribute_to_score_map
        self.class_weights = class_weights

        self.covariates_cols = covariates_cols
        if isinstance(self.covariates_cols, str):
            self.covariates_cols = [self.covariates_cols]

        # ----------------------------
        # If the covariates columns include the partition attribute, remove it.
        # E.g. if partition attribute is sex, we don't want to include sex as a covariate anymore.

        if self.covariates_cols is not None:
            self.covariates_cols = [
                c for c in self.covariates_cols if c != self.partition_attribute
            ]

        # ----------------------------

        # ordered mapping
        self.attribute_values = (
            list(attribute_to_score_map.keys()) if attribute_to_score_map else None
        )
        self.score_cols = (
            [attribute_to_score_map[k] for k in self.attribute_values]
            if attribute_to_score_map
            else None
        )

        # data
        self.phenotype = None
        self.input_data = None
        self.data_scaler = None
        self.family = None

        # model(s)
        self.reg_model = None

        # hyperparams
        self._penalty_type = penalty_type
        self._penalty = penalty
        self._add_intercept = add_intercept

        # ---------------------------
        # Load data
        if prs_dataset is not None:
            if standardize_data:
                prs_dataset.standardize_data()
                self.data_scaler = copy.deepcopy(prs_dataset.scaler)

            self.phenotype = prs_dataset.get_phenotype().reshape(-1, 1)
            self.input_data = self._extract_input_data(prs_dataset)
            self.family = prs_dataset.phenotype_likelihood

        # ---------------------------
        # Build base model
        if self.input_data is not None:
            if self.family == "gaussian":
                if self._penalty_type == "l1":
                    base = Lasso(alpha=self._penalty, fit_intercept=add_intercept)
                elif self._penalty_type == "l2":
                    base = Ridge(alpha=self._penalty, fit_intercept=add_intercept)
                elif self._penalty_type == "elasticnet":
                    base = ElasticNet(alpha=self._penalty, fit_intercept=add_intercept)
                else:
                    base = LinearRegression(fit_intercept=add_intercept)
            else:
                C = self._penalty if (self._penalty and self._penalty > 0.0) else np.inf
                base = LogisticRegression(
                    fit_intercept=add_intercept,
                    class_weight=self.class_weights,
                    penalty=self._penalty_type,
                    C=C,
                    max_iter=1000,
                )

            # one model per attribute value
            self.reg_model = {k: copy.deepcopy(base) for k in self.attribute_values}

    # ---------------------------------------------------------------------
    def _extract_input_data(self, prs_dataset, scaler=None):
        attr = prs_dataset.get_data_columns([self.partition_attribute]).reshape(-1)
        prs_data = prs_dataset.get_data_columns(self.score_cols, scaler=scaler)

        if self.covariates_cols is not None:
            covariates = prs_dataset.get_data_columns(
                self.covariates_cols, scaler=scaler
            )
        else:
            covariates = None

        mapped_values = set(self.attribute_to_score_map.keys())

        input_data = {}
        for j, val in enumerate(self.attribute_values):
            mask = attr == val
            if mask.sum() == 0:
                continue

            prs_col = prs_data[:, j].reshape(-1, 1)

            parts = []
            if covariates is not None:
                parts.append(covariates)
            parts.append(prs_col)

            input_data[val] = {
                "data": np.concatenate(parts, axis=1),
                "keep_samples": mask,
            }

        # store global valid mask
        valid_mask = np.array([x in mapped_values for x in attr], dtype=bool)

        return {"groups": input_data, "valid_mask": valid_mask}

    # ---------------------------------------------------------------------
    def fit(self):
        assert self.input_data is not None
        assert self.phenotype is not None

        for val, dat in self.input_data["groups"].items():
            keep = dat["keep_samples"]
            self.reg_model[val].fit(
                dat["data"][keep],
                self.phenotype[keep],
            )

        return self

    # ---------------------------------------------------------------------
    def predict(self, prs_dataset=None):
        if prs_dataset is None:
            input_data = self.input_data
        else:
            input_data = self._extract_input_data(prs_dataset, scaler=self.data_scaler)

        N = next(iter(input_data["groups"].values()))["data"].shape[0]
        pred = np.full(N, np.nan)

        for val, dat in input_data["groups"].items():
            model = self.reg_model.get(val)
            if model is None:
                continue

            keep = dat["keep_samples"]
            X = dat["data"][keep]

            if self.family == "gaussian":
                pred[keep] = model.predict(X).flatten()
            else:
                pred[keep] = model.predict_proba(X)[:, 1]

        return pred

    # ---------------------------------------------------------------------
    def predict_prs(self, prs_dataset=None, logit_scale=False):
        if prs_dataset is None:
            input_data = self.input_data
        else:
            input_data = self._extract_input_data(prs_dataset, scaler=self.data_scaler)

        N = next(iter(input_data["groups"].values()))["data"].shape[0]
        pred = np.full(N, np.nan)

        for val, dat in input_data["groups"].items():
            mdl = self.reg_model.get(val)
            if mdl is None:
                continue

            keep = dat["keep_samples"]
            prs_col = dat["data"][keep, -1]

            coef = np.asarray(mdl.coef_).ravel()
            lin = prs_col * coef[-1]

            if self.family == "gaussian":
                pred[keep] = lin
            else:
                pred[keep] = lin if logit_scale else expit(lin)

        return pred

    # ---------------------------------------------------------------------
    def get_coefficients(self):
        all_coefs = {}
        for k, v in self.reg_model.items():
            coefs = {"Intercept": v.intercept_}
            if self.covariates_cols is not None:
                coefs["Covariates"] = v.coef_.flatten()[: len(self.covariates_cols)]
                coefs["PRS"] = v.coef_.flatten()[len(self.covariates_cols) :]
            else:
                coefs["PRS"] = v.coef_.flatten()
            all_coefs[k] = coefs
        return all_coefs

    # ---------------------------------------------------------------------
    def save(self, output_file):
        try:
            for _, m in self.reg_model.items():
                check_is_fitted(m)
        except NotFittedError:
            raise NotFittedError("Call `.fit()` before saving.")

        with open(output_file, "wb") as outf:
            pickle.dump(
                [
                    self.reg_model,
                    self.partition_attribute,
                    self.attribute_to_score_map,
                    self.attribute_values,
                    self.score_cols,
                    self.covariates_cols,
                    self.data_scaler,
                    self.family,
                ],
                outf,
            )


class AncestrySpecificMultiPRS(object):
    """
    Fit a separate (multi-PRS + covariates) model per ancestry group.
    Each ancestry-specific model is trained on the same input columns
    but fitted using sample weights defined in ancestry weight columns.
    Predictions for an individual are combined across ancestries by
    weighting ancestry-specific predictions by the corresponding ancestry weight.
    """

    def __init__(
        self,
        prs_dataset=None,
        expert_cols=None,
        covariates_cols=None,
        expert_ancestry_map=None,
        class_weights=None,
        add_intercept=True,
        standardize_data=True,
        penalty_type=None,
        penalty=0.0,
    ):
        # ---------------------------------------------------------------------
        # Sanity checks
        assert penalty >= 0.0
        if penalty > 0.0:
            assert penalty_type is not None

        if prs_dataset is not None and expert_cols is not None:
            assert expert_ancestry_map is not None

        # ---------------------------------------------------------------------
        # Store config
        self.expert_cols = expert_cols
        if self.expert_cols is not None and isinstance(self.expert_cols, str):
            self.expert_cols = [self.expert_cols]

        self.covariates_cols = covariates_cols
        if self.covariates_cols is not None and isinstance(self.covariates_cols, str):
            self.covariates_cols = [self.covariates_cols]

        self.expert_ancestry_map = expert_ancestry_map
        self.ancestry_weights_cols = None

        if self.expert_ancestry_map is not None:
            # Keep only expert cols present in the map:
            self.expert_cols = [
                c for c in self.expert_cols if c in self.expert_ancestry_map
            ]
            self.ancestry_weights_cols = [
                self.expert_ancestry_map[c] for c in self.expert_cols
            ]
            self.expert_ancestry_map = {
                k: v
                for k, v in self.expert_ancestry_map.items()
                if k in self.expert_cols
            }
            assert len(self.expert_cols) > 0
            assert len(self.ancestry_weights_cols) > 0

        # Data placeholders
        self.phenotype = None
        self.data_scaler = None
        self.input_data = None  # will be dict keyed by ancestry
        self.family = None
        self.reg_model = None  # dict mapping ancestry -> model

        # model hyperparams to instantiate per-ancestry
        self._penalty_type = penalty_type
        self._penalty = penalty
        self._add_intercept = add_intercept
        self._class_weights = class_weights

        # Extract data if dataset provided
        if prs_dataset is not None:
            if standardize_data:
                prs_dataset.standardize_data()
                self.data_scaler = copy.deepcopy(prs_dataset.scaler)

            self.phenotype = prs_dataset.get_phenotype().reshape(-1, 1)
            self.input_data = self._extract_input_data(prs_dataset)

            # determine family
            self.family = prs_dataset.phenotype_likelihood

            # instantiate a model template and then copy per ancestry
            template = self._build_model_template()
            # create per-ancestry models
            self.reg_model = {
                k: copy.deepcopy(template) for k in self.input_data.keys()
            }

    @classmethod
    def from_saved_model(cls, param_file):
        model = cls()
        with open(param_file, "rb") as pf:
            (
                model.reg_model,
                model.expert_cols,
                model.expert_ancestry_map,
                model.ancestry_weights_cols,
                model.covariates_cols,
                model.data_scaler,
                model.family,
            ) = pickle.load(pf)
        return model

    @property
    def N(self):
        if self.phenotype is not None:
            return self.phenotype.shape[0]

    @property
    def K(self):
        if self.expert_cols is not None:
            return len(self.expert_cols)

    @property
    def C(self):
        if self.covariates_cols is not None:
            return len(self.covariates_cols)

    def _build_model_template(self):
        """
        Build a regression estimator according to family / penalty settings.
        """
        if self.family == "gaussian":
            if self._penalty_type == "l1":
                return Lasso(alpha=self._penalty, fit_intercept=self._add_intercept)
            elif self._penalty_type == "l2":
                return Ridge(alpha=self._penalty, fit_intercept=self._add_intercept)
            elif self._penalty_type == "elasticnet":
                return ElasticNet(
                    alpha=self._penalty, fit_intercept=self._add_intercept
                )
            else:
                return LinearRegression(fit_intercept=self._add_intercept)
        else:
            # for logistic, sklearn uses C as inverse-regularization; if penalty=0 -> use inf
            C = self._penalty if (self._penalty and self._penalty > 0.0) else np.inf
            return LogisticRegression(
                fit_intercept=self._add_intercept,
                class_weight=self._class_weights,
                penalty=self._penalty_type,
                C=C,
                max_iter=1000,
            )

    def _extract_input_data(self, prs_dataset, scaler=None):
        """
        Build per-ancestry datasets. For each ancestry weight column,
        we keep:
            - data: [covariates..., all expert PRS columns]
            - keep_samples: boolean mask where ancestry weight > 0 and overall has weight
            - weights: the ancestry weight values (for sample_weight when fitting and for mixing)
        """
        # ancestry weight columns (matrix N x M)
        anc_w = prs_dataset.get_data_columns(self.ancestry_weights_cols)
        # PRS matrix (N x K)
        prs_data = prs_dataset.get_data_columns(self.expert_cols, scaler=scaler)
        # covariates if present
        cov_data = (
            prs_dataset.get_data_columns(self.covariates_cols, scaler=scaler)
            if self.covariates_cols is not None
            else None
        )

        input_data = {}
        general_keep = anc_w.sum(axis=1) > 0.5  # require some ancestry support
        for idx, anc_col in enumerate(self.ancestry_weights_cols):
            keep = (anc_w[:, idx] > 0) & general_keep
            if keep.sum() > 0:
                parts = []
                if cov_data is not None:
                    parts.append(cov_data)
                parts.append(prs_data)  # keep all expert PRSs as inputs
                data = np.concatenate(parts, axis=1)
                input_data[anc_col] = {
                    "data": data,
                    "keep_samples": keep,
                    "weights": anc_w[:, idx],
                }
        return input_data

    def predict(self, prs_dataset=None, logit_scale=False):
        """
        Predict combined PRS outcome. If prs_dataset is provided, data will be
        extracted using the stored scaler.
        """
        if prs_dataset is None:
            input_data = self.input_data
        else:
            input_data = self._extract_input_data(prs_dataset, scaler=self.data_scaler)

        # combine ancestry-specific predictions weighted by ancestry weights
        pred = None
        keep_overall = None
        for anc, dat in input_data.items():
            model = self.reg_model.get(anc)
            if model is None:
                continue

            X = dat["data"]
            if self.family == "gaussian":
                pred_anc = model.predict(X).flatten()
            else:
                proba = model.predict_proba(X)[:, 1]
                pred_anc = logit(proba) if logit_scale else proba

            w = dat["weights"]
            if pred is None:
                pred = pred_anc * w
                keep_overall = dat["keep_samples"]
            else:
                pred += pred_anc * w
                keep_overall = keep_overall | dat["keep_samples"]

        if pred is None:
            return None

        # mask samples with no ancestry support
        pred[~keep_overall] = np.nan
        # if binary and user requested probabilities on original scale, apply expit
        if (self.family != "gaussian") and (not logit_scale):
            return expit(pred)
        return pred

    def get_coefficients(self):
        assert self.reg_model is not None
        all_coefs = {}
        for anc, model in self.reg_model.items():
            coefs = {"Intercept": model.intercept_}
            if self.expert_cols is not None:
                if self.covariates_cols is not None:
                    coefs["Covariates"] = model.coef_.flatten()[: self.C]
                    coefs["PRS"] = model.coef_.flatten()[self.C :]
                else:
                    coefs["PRS"] = model.coef_.flatten()
            else:
                coefs["Covariates"] = model.coef_.flatten()
            all_coefs[anc] = coefs
        return all_coefs

    def fit(self):
        """
        Fit each ancestry-specific model using sample weights from ancestry_weights_cols.
        """
        assert self.input_data is not None
        assert self.phenotype is not None

        for anc, dat in self.input_data.items():
            model = self.reg_model[anc]
            X = dat["data"][dat["keep_samples"]]
            y = self.phenotype[dat["keep_samples"]]
            w = dat["weights"][dat["keep_samples"]]
            # sklearn estimators accept sample_weight for .fit
            # for LinearRegression sample_weight is supported (since sklearn 0.24+)
            model = model.fit(X, y.ravel(), sample_weight=w)
            self.reg_model[anc] = model

        return self

    def save(self, output_file):
        """
        Save the per-ancestry models and minimal metadata.
        """
        try:
            for _, model in self.reg_model.items():
                check_is_fitted(model)
        except NotFittedError:
            raise NotFittedError(
                "The model has not been fitted yet. Call `.fit()` first."
            )

        with open(output_file, "wb") as outf:
            pickle.dump(
                [
                    self.reg_model,
                    self.expert_cols,
                    self.expert_ancestry_map,
                    self.ancestry_weights_cols,
                    self.covariates_cols,
                    self.data_scaler,
                    self.family,
                ],
                outf,
            )
