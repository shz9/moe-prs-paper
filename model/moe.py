import numpy as np
import pandas as pd
from scipy.special import softmax, logsumexp, log_softmax, expit
from scipy.linalg import lstsq
from scipy.optimize import minimize
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
import copy
import pickle
from joblib import Parallel, delayed


def _concat_zero(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)


class ParameterTracker(object):

    def __init__(self, params=None, objective=None):

        self.current_params = params
        self.best_params = params
        self.best_objective = objective
        self.curr_iter = 0
        self.best_iter = 0

    def all_close(self, new_params, atol=1e-6):
        if isinstance(new_params, dict):
            return all([np.allclose(self.current_params[k], v, atol=atol) for k, v in new_params.items()])
        else:
            return np.allclose(self.current_params, new_params)

    def add(self, new_params, new_objective):

        self.curr_iter += 1

        if isinstance(new_params, dict):
            self.current_params = {k: v.copy() for k, v in new_params.items()}
        else:
            self.current_params = new_params.copy()

        if self.best_objective is None or new_objective < self.best_objective:

            self.best_objective = new_objective
            self.best_iter = self.curr_iter

            if isinstance(new_params, dict):
                self.best_params = {k: v.copy() for k, v in new_params.items()}
            else:
                self.best_params = new_params.copy()


class MoEPRS(object):

    def __init__(self,
                 prs_dataset=None,
                 expert_cols=None,
                 gate_input_cols=None,
                 expert_covariates_cols=None,
                 gate_add_intercept=True,
                 expert_add_intercept=True,
                 standardize_data=True,
                 gate_penalty=0.,
                 expert_penalty=0.,
                 class_weights=None,
                 fix_residuals=False,
                 optimizer='L-BFGS-B',
                 batch_size=None,
                 n_jobs=1):

        """
        :param prs_dataset: An instance of `PRSDataset` with containing the data for training the model.
        :param expert_cols: The names of the columns to be used as expert predictions.
        :param gate_input_cols: The names of the columns to be used as inputs for the gating model.
        :param expert_covariates_cols: The names of the columns to be used as covariates for the experts (optional).
        :param gate_add_intercept: If True, add an intercept term to the gating model
        :param expert_add_intercept: If True, add an intercept term to the experts
        :param gate_penalty: The penalty term for the gating model
        :param expert_penalty: The penalty term for the experts
        :param fix_residuals: If True, fix the residual variance for each expert to 1.
        :param optimizer: The optimizer to use for fitting the gate parameters in the M-Step.
        Supported options: `lstsq`, `L-BFGS-B`, `switch`.
        :param batch_size: The batch size to use for the L-BFGS-B optimizer.
        :param n_jobs: The number of jobs to use for parallel processing.
        """

        # -------------------------------------------------------------------------
        # Sanity checks:

        assert optimizer in ('lstsq', 'L-BFGS-B', 'switch')
        assert gate_penalty >= 0. and expert_penalty >= 0.

        # -------------------------------------------------------------------------
        # Process model options/optimization parameters:

        self.gate_penalty = gate_penalty
        self.expert_penalty = expert_penalty

        # Intercept information:
        self.gate_add_intercept = gate_add_intercept
        self.expert_add_intercept = expert_add_intercept

        # M-Step optimizer information:
        # If the user selected to switch between optimizers, we flag this here:
        # and start with the faster least squares optimizer.
        self.switch_optimizer = optimizer == 'switch'

        if optimizer == 'switch':
            self.optimizer = 'lstsq'
        else:
            self.optimizer = optimizer

        # Whether to fix the residual variance during optimization
        # (only relevant for continuous phenotypes)
        self.fix_residuals = fix_residuals

        self.n_jobs = n_jobs
        self.batch_size = batch_size

        # -------------------------------------------------------------------------
        # Process / extract training data:

        # Initialize the quantities used to hold the data:
        self.gate_input = None
        self.phenotype = None
        self.expert_predictions = None
        self.expert_covariates = None
        self._expert_loss = None
        self._sq_gate_input = None

        # Initialize the data scaler:
        self.data_scaler = None

        # Initialize / store the names of the columns to be used as inputs:
        self.gate_input_cols = gate_input_cols
        self.expert_covariates_cols = expert_covariates_cols
        self.expert_cols = expert_cols

        if prs_dataset is not None:

            # If standardize_data is True, standardize the training data:
            if standardize_data:
                prs_dataset.standardize_data()
                self.data_scaler = copy.deepcopy(prs_dataset.scaler)

            # Process the phenotype data:
            self.phenotype = prs_dataset.get_phenotype().reshape(-1, 1)

            # Process the inputs for the gating model:
            if self.gate_input_cols is not None:
                self.gate_input = prs_dataset.get_data_columns(self.gate_input_cols,
                                                               add_intercept=self.gate_add_intercept)
            elif self.gate_add_intercept:
                self.gate_input = np.ones((prs_dataset.N, 1))
            else:
                raise ValueError("No covariates provided for the gating model.")

            # This quantity will be used in estimating the gating parameters:
            self._sq_gate_input = self.gate_input.T.dot(self.gate_input)

            # Process the expert predictions:
            self.expert_predictions = prs_dataset.get_data_columns(self.expert_cols)

            # Process expert covariates:
            if self.expert_covariates_cols is not None:
                self.expert_covariates = prs_dataset.get_data_columns(self.expert_covariates_cols,
                                                                      add_intercept=self.expert_add_intercept)
            elif self.expert_add_intercept:
                self.expert_covariates = np.ones((prs_dataset.N, 1))

        # -------------------------------------------------------------------------
        # Initialize containers for model parameters:

        # The gating parameters are C x (K - 1), where C is the number of covariates
        # and K is the number of experts. We have K - 1 because of the constraint
        # that the expert weights have to sum to 1.
        self.gate_params = None
        self.expert_params = None

        # Initialize model parameters:

        self.log_w = None  # Log of the expert weights
        self.log_resp = None  # Log of the expert responsibilities
        self.log_resid = None  # The residual for each expert

        # Determine the family for the regression model:
        if prs_dataset is not None:
            self.family = prs_dataset.phenotype_likelihood
        else:
            self.family = None

        # Process class weights (if any) for the binomial family:
        if self.family is not None and self.family == 'binomial':
            if class_weights is None:
                self.class_weights = np.array([1., 1.])
            elif class_weights == 'balanced':
                self.class_weights = self.N / (2 * np.bincount(self.phenotype.flatten().astype(np.int32)))
            else:
                self.class_weights = np.array(class_weights)
        else:
            self.class_weights = None

        # Initialize container to fit weighted linear models in the M Step:

        self.wl_model = None

        # -------------------------------------------------------------------------
        # Initialize containers for keep track of objective / parameter values:

        self.history = None
        self.param_tracker = ParameterTracker()

    @classmethod
    def from_saved_model(cls, param_file):
        """
        Initialize a MoEPRS model from a saved model file.
        """
        model = cls()

        with open(param_file, "rb") as pf:
            (model.gate_params,
             model.expert_params,
             model.gate_add_intercept,
             model.expert_add_intercept,
             model.gate_input_cols,
             model.expert_cols,
             model.expert_covariates_cols,
             model.family,
             model.data_scaler) = pickle.load(pf)

        return model

    def initialize(self, param_0=None, init_history=True):
        """
        Initialize the model parameters and optimization history.
        This function initializes the gating model parameters and the expert
        parameters.

        :param param_0: A dictionary with the initial values for the model parameters (mainly gate parameters).
        :param init_history: If True, initialize the history of the optimization.
        """

        if init_history:
            self.history = {
                'NCLL': [],
                'Weighted Loss': [],
                'Ensemble Loss': [],
                'Expert Losses': [],
                'Weighted NLL': [],
                'Gate Loss': [],
                'Model Weights': []
            }

        if param_0 is not None and 'gate_params' in param_0:
            assert param_0['gate_params'].shape == (self.gate_dim, self.K - 1)
            self.gate_params = param_0['gate_params']
        else:
            self.gate_params = np.random.normal(scale=0.01, size=(self.gate_dim, self.K - 1))

        self.log_w = self.predict_proba(log=True)
        self.log_resp = self.log_w.copy()

        if self.expert_covariates is not None:
            self.expert_params = np.random.normal(scale=0.01, size=(self.K, self.expert_dim))

        self.update_expert_losses()

        if self.family == 'gaussian':
            if self.fix_residuals:
                self.log_resid = np.zeros(self.K)
            else:
                self.update_residuals()

            if self.expert_penalty > 0.:
                self.wl_model = Ridge(fit_intercept=False, alpha=self.expert_penalty)
            else:
                self.wl_model = LinearRegression(fit_intercept=False)
        else:
            if self.expert_penalty > 0.:
                self.wl_model = LogisticRegression(fit_intercept=False,
                                                   class_weight=dict(zip([0., 1.], self.class_weights)),
                                                   C=1. / self.expert_penalty,
                                                   penalty='l2')
            else:
                self.wl_model = LogisticRegression(fit_intercept=False,
                                                   class_weight=dict(zip([0., 1.], self.class_weights)),
                                                   penalty=None)

    @property
    def n_params(self):
        """
        The number of parameters in the model
        """
        return self.gate_params.size + (self.expert_params.size if self.expert_params is not None else 0)

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
    def gate_dim(self):
        """
        The dimension of the input to the gating model
        """
        if self.gate_input is not None:
            return len(self.gate_input_cols) + (1 if self.gate_add_intercept else 0)

    @property
    def expert_dim(self):
        """
        The dimension of the input for each expert
        """
        d = 1
        if self.expert_covariates_cols is not None:
            d += len(self.expert_covariates_cols)
        if self.expert_add_intercept:
            d += 1
        return d

    @property
    def expert_responsibility(self):
        if self.log_resp is not None:
            return np.exp(self.log_resp)

    def weighted_loss(self, axis=None):
        return (1. / self.N) * (np.exp(self.log_w) * self._expert_loss).sum(axis=axis)

    def weighted_nll(self, axis=None):
        return (-1. / self.N) * (self.expert_responsibility * self.ll()).sum(axis=axis)

    def gate_loss(self, axis=None):
        return (-1. / self.N) * (self.expert_responsibility * self.log_w).sum(axis=axis)

    def ensemble_loss(self):

        preds = self.predict()
        phenotype = self.phenotype.flatten()

        if self.family == 'gaussian':
            return np.mean((phenotype - preds) ** 2)
        else:
            preds = np.clip(preds, a_min=1e-6, a_max=1. - 1e-6)
            return np.mean(-(self.class_weights[1] * phenotype * np.log(preds) +
                             self.class_weights[0] * (1. - phenotype) * np.log(1. - preds)))

    def objective(self):
        return self.complete_nll()

    def complete_nll(self):

        expert_resp = self.expert_responsibility
        w_loss = -(1. / self.N) * np.sum(expert_resp * (self.log_w + self.ll()))

        if self.gate_penalty > 0.:
            # Add penalty term for the gating model:
            # Here, we scale the penalty by the sample size because we divide by N
            # later on.
            w_loss += self.gate_penalty * (self.gate_params ** 2).sum()

        if self.expert_params is not None and self.expert_penalty > 0.:
            # Add penalty term for the experts:
            # np.dot(expert_resp.sum(axis=0), (self.expert_params ** 2).sum(axis=1))
            w_loss += self.expert_penalty * (self.expert_params ** 2).sum()

        return w_loss

    def ll(self, axis=None):
        """
        Return the log-likelihood for each expert and individual
        """

        if self.family == 'gaussian':
            ll = -0.5 * (np.log(2. * np.pi) + self.log_resid +
                         np.exp(-self.log_resid) * self._expert_loss)
        else:
            ll = -self._expert_loss

        if axis is not None:
            return ll.sum(axis=axis)
        else:
            return ll

    def update_expert_losses(self):

        preds = self.get_scaled_predictions()

        if self.family == 'gaussian':
            self._expert_loss = (self.phenotype - preds) ** 2
        else:
            preds = np.clip(preds, a_min=1e-6, a_max=1. - 1e-6)
            self._expert_loss = -(self.class_weights[1] * self.phenotype * np.log(preds) +
                                  self.class_weights[0] * (1. - self.phenotype) * np.log(1. - preds))

    def get_scaled_predictions(self, prs_dataset=None):
        """
        Get the scaled predictions for each expert. Specifically, we scale the prediction of
        each expert by the parameters of the linear model present in `expert_params`. If we define
        expert i as PRS_i, then the scaled prediction for expert i is given by:

        PRS_i(scaled) = \alpha_i + \sum_{c=1}^{C} \beta_{c}^{(i)} X_{c} + \beta_{0}^{(i)} PRS_i

        Where C is the number of covariates, \alpha_i is the intercept term for expert i,
        and \beta_{0}^{(i)} is the coefficient for the PRS itself. The remaining coefficients
        are for the covariates in the model (if present).

        :param prs_dataset: An instance of `PRSDataset` with the data to use for scaling the predictions.
        """

        assert self.expert_cols is not None

        # Process the expert predictions:

        if prs_dataset is None:
            expert_predictions = self.expert_predictions
        else:
            expert_predictions = prs_dataset.get_data_columns(self.expert_cols, scaler=self.data_scaler)

        assert expert_predictions is not None

        if self.expert_params is not None:

            if prs_dataset is None:
                expert_covariates = self.expert_covariates
            else:
                if self.expert_covariates_cols is not None:
                    expert_covariates = prs_dataset.get_data_columns(self.expert_covariates_cols,
                                                                     add_intercept=self.expert_add_intercept,
                                                                     scaler=self.data_scaler)
                elif self.expert_add_intercept:
                    expert_covariates = np.ones((prs_dataset.N, 1))

            # Step 1: Separate the last parameter and the remaining parameters in expert_params
            scale_params = self.expert_params[:, -1:]  # Shape: Kx1
            shift_params = self.expert_params[:, :-1]  # Shape: KxC

            # Step 2: Calculate the dot product of covariates and sp_remaining
            dot_product = expert_covariates.dot(shift_params.T)  # Shape: NxK

            # Step 3: Multiply predictions by sp_last
            predictions_mult = expert_predictions * scale_params.T  # Shape: NxK

            # Step 4: Add the results of step 2 and step 3
            expert_predictions = dot_product + predictions_mult  # Shape: NxK

        if self.family == 'gaussian':
            return expert_predictions
        else:
            return expit(expert_predictions)

    def e_step(self):
        """
        The Expectation step of the EM algorithm. This function computes
        the log of the expert responsibilities.
        """
        self.log_resp = log_softmax(self.log_w + self.ll(), axis=-1)

    def m_step(self):
        """
        The maximization step of the EM algorithm.
        Given the expert responsibilities computed in the E-Step, update the
        gating model parameters and the expert parameters. For models with Gaussian likelihoods,
        we also update the residuals of the model.
        """

        # (1) Update the parameters of the gating model:

        # Compute expert responsibilities:
        expert_resp = self.expert_responsibility

        if self.optimizer == 'lstsq':

            H = np.clip(self.log_resp[:, :-1] - self.log_resp[:, -1:],
                        a_min=-40., a_max=40.)
            self.gate_params = lstsq((1. / self.N) * (self._sq_gate_input +
                                                      self.gate_penalty * np.identity(self.gate_dim)),
                                     (1. / self.N) * self.gate_input.T.dot(H))[0]

        elif self.optimizer == 'L-BFGS-B':

            if self.batch_size is not None:
                idx = np.random.choice(self.N, size=self.batch_size)
                gate_input = self.gate_input[idx, :]
                selected_expert_resp = expert_resp[idx, :]
            else:
                gate_input = self.gate_input
                selected_expert_resp = expert_resp

            def local_objective(gparams):

                log_g = log_softmax(
                    _concat_zero(gate_input.dot(gparams.reshape(self.gate_dim, self.K - 1))),
                    axis=-1
                )

                # Scale the loss by 1/sample_size for numerical stability?
                loss = -(1. / self.N) * np.sum(selected_expert_resp * log_g)
                grad = (1. / self.N) * gate_input.T.dot(np.exp(log_g[:, :-1]) - selected_expert_resp[:, :-1]).flatten()

                if self.gate_penalty > 0:
                    loss += self.gate_penalty * (gparams ** 2).sum()
                    grad += 2. * self.gate_penalty * gparams

                return loss, grad

            self.gate_params = minimize(local_objective,
                                        self.gate_params.flatten(),
                                        jac=True,
                                        method='L-BFGS-B').x.reshape(self.gate_dim, self.K - 1)

        # -------------------------------------------------------------------------

        # (2) Update the parameters of the experts:

        if self.expert_params is not None:

            def fit_model(i):
                model = copy.deepcopy(self.wl_model)
                x = np.concatenate([self.expert_covariates,
                                    self.expert_predictions[:, i, None]], axis=1)

                model.fit(x, self.phenotype.flatten(), sample_weight=expert_resp[:, i])
                return model.coef_

            self.expert_params = np.array(Parallel(n_jobs=self.n_jobs)(delayed(fit_model)(i)
                                                                       for i in range(len(self.expert_params))))

        # -------------------------------------------------------------------------

        # (3) Given the updated parameters, re-compute the expert weights and losses:
        self.log_w = self.predict_proba(log=True)
        self.update_expert_losses()

        # -------------------------------------------------------------------------
        # (4) Update the residuals (for Gaussian likelihoods):
        self.update_residuals()

    def update_residuals(self):
        """
        Update the residual variance for the experts (applicable when
        the phenotype likelihood is Gaussian).
        """

        if not self.fix_residuals and self.family == 'gaussian':
            self.log_resid = (
                    logsumexp(self.log_resp + np.log(self._expert_loss), axis=0) -
                    logsumexp(self.log_resp, axis=0)
            )

    def predict_proba(self, prs_dataset=None, log=False):
        """
        Predict the expert weights based on an input set of covariates.
        :param prs_dataset: An instance of `PRSDataset` containing the covariates to use as input
        for the gating model.
        :param log: If True, return log probabilities.
        """

        assert self.gate_input_cols is not None

        if prs_dataset is None:
            gate_input = self.gate_input
        else:
            gate_input = prs_dataset.get_data_columns(self.gate_input_cols,
                                                      add_intercept=self.gate_add_intercept,
                                                      scaler=self.data_scaler)

        assert gate_input is not None

        logits = _concat_zero(gate_input.dot(self.gate_params))

        if log:
            return log_softmax(logits, axis=-1)
        else:
            return softmax(logits, axis=-1)

    def predict(self, prs_dataset=None):
        """
        Predict for new samples
        """

        scaled_pred = self.get_scaled_predictions(prs_dataset)

        return (self.predict_proba(prs_dataset) * scaled_pred).sum(axis=1)

    def get_model_parameters(self):

        params = {
            'gate_params': pd.DataFrame(self.gate_params,
                                        index=[[], ['Intercept']][self.gate_add_intercept] + self.gate_input_cols,
                                        columns=self.expert_cols[:-1])
        }

        if self.expert_params is not None:
            columns = []
            if self.expert_add_intercept:
                columns.append('Intercept')
            columns += self.expert_covariates_cols or []
            columns += ['PRS']

            params['expert_params'] = pd.DataFrame(self.expert_params,
                                                   index=self.expert_cols,
                                                   columns=columns)
        if self.log_resid is not None:
            params['log_resid'] = pd.DataFrame(self.log_resid,
                                               columns=['Log Residual Variance'],
                                               index=self.expert_cols)

        return params

    def fit(self,
            n_iter=1000,
            n_restarts=1,
            param_0=None,
            patience=5,
            atol=1e-4,
            objective='ncll'):

        objective = objective.lower()

        if objective == 'ncll':
            objective = 'NCLL'
        elif objective == 'ensemble_loss':
            objective = 'Ensemble Loss'
        else:
            raise ValueError("Objective must be one of 'ncll' or 'ensemble_loss'")

        pbar = tqdm(range(n_restarts * n_iter))

        restart = True
        curr_iter = 0
        self.initialize(param_0=param_0)

        for _ in pbar:

            if restart or curr_iter >= n_iter:
                # Decrease the number of restarts:
                n_restarts -= 1
                # If the number of restarts < 0, break out of the loop:
                if n_restarts < 0:
                    break

                # Otherwise, re-initialize and start over:
                curr_iter = 0
                patience_r = patience
                restart = False
                self.initialize(param_0=param_0, init_history=False)

            self.e_step()
            self.m_step()

            self.history['NCLL'].append(self.complete_nll())
            self.history['Weighted Loss'].append(self.weighted_loss())
            self.history['Weighted NLL'].append(self.weighted_nll())
            self.history['Gate Loss'].append(self.gate_loss())
            self.history['Ensemble Loss'].append(self.ensemble_loss())
            self.history['Expert Losses'].append(self.weighted_nll(axis=0))
            self.history['Model Weights'].append(self.expert_responsibility.mean(axis=0))

            if curr_iter > 0:
                if np.allclose(self.history[objective][-1], self.history[objective][-2], atol=atol):
                    print(f"Converged at iteration {curr_iter}")
                    restart = True
                elif self.param_tracker.all_close(self.get_model_parameters(), atol=atol):
                    print(f"Converged at iteration {curr_iter}")
                    restart = True
                elif patience_r < 1:
                    print("Model is no longer improving; Breaking...")
                    restart = True
                elif self.history['NCLL'][-1] > self.history['NCLL'][-2]:
                    patience_r -= 1
                    if self.switch_optimizer and self.optimizer == 'lstsq':
                        print(f"> Iteration {curr_iter}: Switching optimizer from lstsq to L-BFGS-B.")
                        self.optimizer = 'L-BFGS-B'

            curr_iter += 1
            self.param_tracker.add(self.get_model_parameters(), self.history[objective][-1])
            pbar.set_postfix({'NCLL': self.history['NCLL'][-1],
                              'Loss': self.history['Ensemble Loss'][-1],
                              'Restarts remaining': n_restarts})

        # Retrieve the parameters with the best objective value:
        self.gate_params = self.param_tracker.best_params['gate_params'].values.copy()
        if self.expert_params is not None:
            self.expert_params = self.param_tracker.best_params['expert_params'].values.copy()
        if self.log_resid is not None:
            self.log_resid = self.param_tracker.best_params['log_resid'].values.copy()

        return self

    def save(self, output_file):
        """
        Save the parameters of the model to file.
        """

        if self.gate_params is None:
            raise ValueError("Model has not been fitted yet. Call `.fit() first.")

        with open(output_file, "wb") as outf:
            pickle.dump([
                self.gate_params,
                self.expert_params,
                self.gate_add_intercept,
                self.expert_add_intercept,
                self.gate_input_cols,
                self.expert_cols,
                self.expert_covariates_cols,
                self.family,
                self.data_scaler
            ], outf)


