import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import copy
import pickle


class MultiPRS(object):
    
    def __init__(self,
                 prs_dataset=None,
                 expert_cols=None,
                 covariates_cols=None,
                 class_weights=None,
                 add_intercept=True,
                 standardize_data=True,
                 penalty_type=None,
                 penalty=0.):

        # -------------------------------------------------------------------------
        # Sanity checks:

        assert penalty >= 0.
        if penalty > 0.:
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
        if self.family == 'gaussian':
            if penalty_type == 'l1':
                self.reg_model = Lasso(alpha=penalty, fit_intercept=add_intercept)
            elif penalty_type == 'l2':
                self.reg_model = Ridge(alpha=penalty, fit_intercept=add_intercept)
            elif penalty_type == 'elasticnet':
                self.reg_model = ElasticNet(alpha=penalty, fit_intercept=add_intercept)
            else:
                self.reg_model = LinearRegression(fit_intercept=add_intercept)
        else:
            if penalty == 0.:
                penalty = np.inf

            self.reg_model = LogisticRegression(fit_intercept=add_intercept,
                                                class_weight=class_weights,
                                                penalty=penalty_type,
                                                C=penalty)
        
    @classmethod
    def from_saved_model(cls, param_file):

        model = cls()
        
        with open(param_file, "rb") as pf:
            (model.reg_model, model.expert_cols, model.covariates_cols,
             model.data_scaler, model.family) = pickle.load(pf)

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
    
    def predict(self, prs_dataset=None):

        assert self.input_cols is not None

        if prs_dataset is None:
            input_data = self.input_data
        else:
            input_data = prs_dataset.get_data_columns(self.input_cols,
                                                      scaler=self.data_scaler)

        if self.family == "gaussian":
            return self.reg_model.predict(input_data).flatten()
        else:
            return self.reg_model.predict_proba(input_data)[:, 1]

    def get_coefficients(self):

        assert self.reg_model is not None

        coefs = {'Intercept': self.reg_model.intercept_}

        if self.expert_cols is not None:
            if self.covariates_cols is not None:
                coefs['Covariates'] = self.reg_model.coef_[:self.C]
                coefs['PRS'] = self.reg_model.coef_[self.C:]
            else:
                coefs['PRS'] = self.reg_model.coef_
        else:
            coefs['Covariates'] = self.reg_model.coef_

        return coefs

    def fit(self):
        """
        Fit the model to the training data
        """
        assert self.input_data is not None
        assert self.phenotype is not None

        self.reg_model = self.reg_model.fit(self.input_data, self.phenotype)
    
    def save(self, output_file):
        """
        Save the parameters of the model to file.
        """

        try:
            check_is_fitted(self.reg_model)
        except NotFittedError:
            raise NotFittedError("The model has not been fitted yet. Call `.fit() first.")

        with open(output_file, "wb") as outf:
            pickle.dump([
                self.reg_model,
                self.expert_cols,
                self.covariates_cols,
                self.data_scaler,
                self.family
            ], outf)


class AncestryWeightedPRS(object):
    """
    A class to model the PRS using pre-defined ancestry weights (e.g.
    from gnomAD random forest classifier).
    """

    def __init__(self,
                 prs_dataset=None,
                 expert_cols=None,
                 covariates_cols=None,
                 expert_ancestry_map=None,
                 weighing_scheme='before',
                 class_weights=None,
                 add_intercept=True,
                 standardize_data=True,
                 penalty_type=None,
                 penalty=0.):

        # -------------------------------------------------------------------------
        # Sanity checks:

        assert penalty >= 0.
        if penalty > 0.:
            assert penalty_type is not None

        assert weighing_scheme in ('before', 'after')
        if prs_dataset is not None and expert_cols is not None:
            assert expert_ancestry_map is not None

        # -------------------------------------------------------------------------
        # Process / extract training data:

        # Initialize the weighing scheme:
        self.weighing_scheme = weighing_scheme

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

        # Initialize the map between the expert columns vs. ancestry weights columns:
        self.expert_ancestry_map = expert_ancestry_map
        self.ancestry_weights_cols = None

        if self.expert_ancestry_map is not None:

            # Only keep PRS columns that are present in the mapping:
            self.expert_cols = [col for col in self.expert_cols if col in self.expert_ancestry_map]

            # Get ancestry weights columns that have corresponding PRS columns.
            self.ancestry_weights_cols = [self.expert_ancestry_map[col] for col in self.expert_cols]

            # Update the ancestry:PRS map based on the previous filtering steps:
            self.expert_ancestry_map = {k: v for k, v in self.expert_ancestry_map.items()
                                        if k in self.expert_cols}

            assert len(self.expert_cols) > 1
            assert len(self.ancestry_weights_cols) > 1

        # Extract the data needed to train this model:
        if prs_dataset is not None:

            # If standardize_data is True, standardize the training data:
            if standardize_data:
                prs_dataset.standardize_data()
                self.data_scaler = copy.deepcopy(prs_dataset.scaler)

            # Process the phenotype data:
            self.phenotype = prs_dataset.get_phenotype().reshape(-1, 1)
            # Extract the input data for training the model:
            self.input_data = self._extract_input_data(prs_dataset)

        # -------------------------------------------------------------------------

        # Determine the family of the phenotype:
        if prs_dataset is not None:
            self.family = prs_dataset.phenotype_likelihood
        else:
            self.family = None

        if self.input_data is not None:
            # Initialize the regression model:
            if self.family == 'gaussian':
                if penalty_type == 'l1':
                    self.reg_model = Lasso(alpha=penalty, fit_intercept=add_intercept)
                elif penalty_type == 'l2':
                    self.reg_model = Ridge(alpha=penalty, fit_intercept=add_intercept)
                elif penalty_type == 'elasticnet':
                    self.reg_model = ElasticNet(alpha=penalty, fit_intercept=add_intercept)
                else:
                    self.reg_model = LinearRegression(fit_intercept=add_intercept)
            else:
                if penalty == 0.:
                    penalty = np.inf

                self.reg_model = LogisticRegression(fit_intercept=add_intercept,
                                                    class_weight=class_weights,
                                                    penalty=penalty_type,
                                                    C=penalty)

            # Create a separate model for each ancestry:
            if self.weighing_scheme == 'after':
                self.reg_model = {k: copy.deepcopy(self.reg_model) for k in self.input_data.keys()}

    @classmethod
    def from_saved_model(cls, param_file):

        model = cls()
        with open(param_file, "rb") as pf:
            (model.reg_model, model.expert_cols, model.expert_ancestry_map,
             model.ancestry_weights_cols, model.covariates_cols,
             model.weighing_scheme, model.data_scaler, model.family) = pickle.load(pf)

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

    def _extract_input_data(self, prs_dataset, scaler=None):
        """
        Extract the relevant input data from the PRS dataset and
        depending on the weighing scheme.
        """

        # First, extract the ancestry weight columns:
        ancestry_weights = prs_dataset.get_data_columns(self.ancestry_weights_cols)

        # Extract the PRS columns:
        prs_data = prs_dataset.get_data_columns(self.expert_cols, scaler=scaler)

        # Extract the covariates columns (if requested):
        if self.covariates_cols is not None:
            covariates_data = [prs_dataset.get_data_columns(self.covariates_cols, scaler=scaler)]
        else:
            covariates_data = []

        if self.weighing_scheme == 'before':
            weighted_prs = (prs_data * ancestry_weights).sum(axis=1).reshape(-1, 1)

            return {
                'data': np.concatenate(covariates_data + [weighted_prs], axis=1),
                'keep_samples': ancestry_weights.sum(axis=1) > .5,
                'weights': None
            }
        else:

            # If the weighing scheme is set to `after`, then we need to generate
            # a separate dataset for each ancestry group.
            input_data = {}

            general_keep = ancestry_weights.sum(axis=1) > .5

            for idx, anc_col in enumerate(self.ancestry_weights_cols):
                keep_samples = (ancestry_weights[:, idx] > 0) & general_keep
                if keep_samples.sum() > 0:
                    input_data[anc_col] = {
                        'data': np.concatenate(covariates_data + [prs_data[:, idx].reshape(-1, 1)], axis=1),
                        'keep_samples': keep_samples,
                        'weights': ancestry_weights[:, idx]
                    }
            return input_data

    def predict(self, prs_dataset=None):

        if prs_dataset is None:
            input_data = self.input_data
        else:
            input_data = self._extract_input_data(prs_dataset, scaler=self.data_scaler)

        if self.weighing_scheme == 'before':
            if self.family == "gaussian":
                pred = self.reg_model.predict(input_data['data']).flatten()
            else:
                pred = self.reg_model.predict_proba(input_data['data'])[:, 1]

            pred[~input_data['keep_samples']] = np.nan
        else:
            pred = None
            keep_samples = None
            for anc, dat in input_data.items():
                if self.family == "gaussian":
                    pred_anc = self.reg_model[anc].predict(dat['data']).flatten()
                else:
                    pred_anc = self.reg_model[anc].predict_proba(dat['data'])[:, 1]

                if pred is None:
                    pred = pred_anc*dat['weights']
                    keep_samples = dat['keep_samples']
                else:
                    pred += pred_anc*dat['weights']
                    keep_samples = keep_samples | dat['keep_samples']

            pred[~keep_samples] = np.nan

        return pred

    def get_coefficients(self):

        assert self.reg_model is not None

        if self.weighing_scheme == 'before':
            model = {'All': self.reg_model}
        else:
            model = self.reg_model

        all_coefs = {}

        for k, v in model.items():
            coefs = {'Intercept': v.intercept_}

            if self.expert_cols is not None:
                if self.covariates_cols is not None:
                    coefs['Covariates'] = v.coef_[:self.C]
                    coefs['PRS'] = v.coef_[self.C:]
                else:
                    coefs['PRS'] = v.coef_
            else:
                coefs['Covariates'] = v.coef_

            all_coefs[k] = coefs

        return all_coefs

    def fit(self):
        """
        Fit the model to the training data
        """
        assert self.input_data is not None
        assert self.phenotype is not None

        if self.weighing_scheme == 'before':
            self.reg_model = self.reg_model.fit(self.input_data['data'][self.input_data['keep_samples']],
                                                self.phenotype[self.input_data['keep_samples']])
        else:
            for anc, dat in self.input_data.items():
                # Fit a weighted model:
                self.reg_model[anc] = self.reg_model[anc].fit(dat['data'][dat['keep_samples']],
                                                              self.phenotype[dat['keep_samples']],
                                                              sample_weight=dat['weights'][dat['keep_samples']])

    def save(self, output_file):
        """
        Save the parameters of the model to file.
        """

        try:
            if self.weighing_scheme == 'before':
                check_is_fitted(self.reg_model)
            else:
                for _, model in self.reg_model.items():
                    check_is_fitted(model)
        except NotFittedError:
            raise NotFittedError("The model has not been fitted yet. Call `.fit() first.")

        # TODO
        with open(output_file, "wb") as outf:
            pickle.dump([
                self.reg_model,
                self.expert_cols,
                self.expert_ancestry_map,
                self.ancestry_weights_cols,
                self.covariates_cols,
                self.weighing_scheme,
                self.data_scaler,
                self.family
            ], outf)
