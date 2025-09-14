import numpy as np
import pandas as pd
from scipy.special import softmax, logsumexp, log_softmax, expit, huber
from scipy.linalg import lstsq
from scipy.optimize import minimize
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, HuberRegressor
import copy
import pickle
from joblib import Parallel, delayed


def _concat_zero(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)


class ParameterTracker(object):
    """
    An object to track the objective and parameters of
    the MoE model.
    """

    def __init__(self, params=None, objective=None):

        self.current_params = params
        self.best_params = params
        self.best_objective = objective
        self.curr_iter = 0
        self.best_iter = 0

    def all_close(self, new_params, atol=1e-6, rtol=0.):
        if isinstance(new_params, dict):
            return all([np.allclose(self.current_params[k], v, atol=atol, rtol=rtol)
                        for k, v in new_params.items()])
        else:
            return np.allclose(self.current_params, new_params, atol=atol, rtol=rtol)

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

    def reset(self):

        self.current_params = None
        self.best_params = None
        self.best_objective = None
        self.curr_iter = 0
        self.best_iter = 0




class MoEPRS(object):

    def __init__(self,
                 prs_dataset=None,
                 expert_cols=None,
                 gate_input_cols=None,
                 expert_covariates_cols=None,
                 global_covariates_cols=None,
                 gate_add_intercept=True,
                 expert_add_intercept=True,
                 standardize_data=True,
                 gate_penalty=0.,
                 expert_penalty=0.,
                 loss='infer',
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
        :param gate_add_intercept: If True, add an intercept term to the gating model.
        :param expert_add_intercept: If True, add an intercept term to the experts.
        :param gate_penalty: The penalty term for the gating model.
        :param expert_penalty: The penalty term for the experts.
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
        self.global_covariates = None
        self._expert_loss = None
        self._sq_gate_input = None

        # Initialize the data scaler:
        self.data_scaler = None

        # Initialize / store the names of the columns to be used as inputs:
        self.gate_input_cols = gate_input_cols
        self.expert_covariates_cols = expert_covariates_cols
        self.expert_cols = expert_cols
        self.global_covariates_cols = global_covariates_cols

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

            if self.global_covariates_cols is not None:
                self.global_covariates = prs_dataset.get_data_columns(self.global_covariates_cols,
                                                                      add_intercept=not self.expert_add_intercept)

        # -------------------------------------------------------------------------
        # Initialize containers for model parameters:

        # The gating parameters are C x (K - 1), where C is the number of covariates
        # and K is the number of experts. We have K - 1 because of the constraint
        # that the expert weights have to sum to 1.
        self.gate_params = None
        self.expert_params = None
        self.global_params = None

        # Initialize model parameters:

        self.log_w = None  # Log of the expert weights
        self.log_resp = None  # Log of the expert responsibilities
        self.log_resid = None  # The residual for each expert

        # Determine the type of the regression model:
        if prs_dataset is not None and loss == 'infer':
            if prs_dataset.phenotype_likelihood == 'gaussian':
                self.loss = 'mse'
            else:
                self.loss = 'bce'  # binary cross-entropy
        else:
            self.loss = loss

        # Process class weights (if any) for the binomial family:
        if self.loss == 'bce':
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
             model.global_params,
             model.log_resid,
             model.gate_add_intercept,
             model.expert_add_intercept,
             model.gate_input_cols,
             model.expert_cols,
             model.expert_covariates_cols,
             model.global_covariates_cols,
             model.loss,
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
            self.gate_params = np.zeros(shape=(self.gate_dim, self.K - 1))

        self.log_w = self.predict_proba(log=True)
        self.log_resp = self.log_w.copy()

        if param_0 is not None and 'expert_params' in param_0:
            assert param_0['expert_params'].shape == (self.K, self.expert_dim)
            self.expert_params = param_0['expert_params']
        else:
            self.expert_params = np.random.normal(scale=0.01, size=(self.K, self.expert_dim))

        if self.global_covariates is not None:
            if param_0 is not None and 'global_params' in param_0:
                assert param_0['global_params'].shape == (self.global_covariates.shape[1],)
                self.global_params = param_0['global_params']
            else:
                self.global_params = np.random.normal(scale=0.01, size=self.global_covariates.shape[1])

        # ------------------------------------------
        # Initialize expert-specific hyperparameters:

        self.update_expert_losses()

        # ------------------------------------------
        # Reset the tracker for the best parameters:
        self.param_tracker.reset()

        # ------------------------------------------

        if self.loss == 'mse':
            if self.fix_residuals:
                self.log_resid = np.zeros(self.K)
            else:
                self.update_residuals()

            if self.expert_penalty > 0.:
                self.wl_model = Ridge(fit_intercept=False, alpha=self.expert_penalty)
            else:
                self.wl_model = LinearRegression(fit_intercept=False)
        elif self.loss == 'huber':
            self.wl_model = HuberRegressor(fit_intercept=False, alpha=self.expert_penalty)
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

        n_params = 0

        if self.gate_params is not None:
            n_params += self.gate_params.size

        if self.expert_params is not None:
            n_params += self.expert_params.size

        if self.global_params is not None:
            n_params += self.global_params.size

        return n_params

    @property
    def N(self):
        """
        The number of samples
        """

        assert self.phenotype is not None
        return self.phenotype.shape[0]

    @property
    def K(self):
        """
        The number of experts
        """

        assert self.expert_cols is not None
        return len(self.expert_cols)

    @property
    def gate_dim(self):
        """
        The dimension of the input to the gating model
        """

        if self.gate_input is not None:
            return self.gate_input.shape[1]
        else:
            if self.gate_input_cols is not None:
                return len(self.gate_input_cols) + int(self.gate_add_intercept)
            elif self.gate_add_intercept:
                return 1
            else:
                raise ValueError("The MoE object is not set up properly; Gate dimension could not be inferred.")


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

        assert self.log_resp is not None
        return np.exp(self.log_resp)

    def weighted_loss(self, axis=None):
        return (1. / self.N) * (np.exp(self.log_w) * self._expert_loss).sum(axis=axis)

    def weighted_nll(self, axis=None):
        return (-1. / self.N) * (self.expert_responsibility * self.ll()).sum(axis=axis)

    def _weighted_nll_grad(self):
        """
        Compute the gradient of the current loss w.r.t. global and expert-specific parameters.
        Supports Gaussian, Huber (with per-expert scale), and binary cross-entropy losses.

        Returns
        -------
        grad_global : np.ndarray, shape (M,)
            Gradient w.r.t. global covariates.
        grad_expert : list of np.ndarray
            List of gradients, one per expert. Each has shape (J_k,)
        """

        y_hat = self.get_predictions()        # shape (N, K)
        y = self.phenotype                  # shape (N, 1)
        W = self.expert_responsibility                  # shape (N, K)

        if self.loss == 'mse':
            residuals = y - y_hat
            deltas = - np.exp(-self.log_resid).reshape(1, -1)*residuals

        elif self.loss == 'huber':
            delta = 1.35
            residuals = y - y_hat                                # shape (N, K)

            abs_residuals = np.abs(residuals)
            mask = abs_residuals <= delta

            # Derivative of the Huber loss w.r.t. y_hat, incorporating scaling
            deltas = np.where(
                mask,
                -residuals,                       # quadratic region
                -delta * np.sign(residuals)       # linear region
            )

        elif self.loss == 'bce':
            deltas = y_hat - y  # y_hat already passed through expit

        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

        # --- Weighted deltas ---
        weighted_deltas = W * deltas  # shape (N, K)

        grads = []

        # --- Gradient w.r.t. global weights ---
        if self.global_params is not None:
            grads.append(np.sum(weighted_deltas @ np.ones((self.K, 1)) * self.global_covariates, axis=0))

        # --- Gradient w.r.t. expert-specific weights ---
        if self.expert_params is not None:
            for k in range(self.K):

                if self.expert_covariates is not None:
                    Zk = np.concatenate([self.expert_covariates, self.expert_predictions[:, k, None]], axis=1)
                else:
                    Zk = self.expert_predictions[:, k, None]

                grad_k = weighted_deltas[:, k] @ Zk  # shape (J_k,)
                grads.append(grad_k)

        return (1./self.N) * np.concatenate(grads)


    def gate_loss(self, axis=None):
        """
        The loss for the gating model
        """
        return (-1. / self.N) * (self.expert_responsibility * self.log_w).sum(axis=axis)

    def ensemble_loss(self):
        """
        The ensemble prediction loss
        """

        preds = self.predict()
        phenotype = self.phenotype.flatten()

        if self.loss == 'mse':
            return np.mean((phenotype - preds) ** 2)
        elif self.loss == 'huber':
            return np.mean(huber(1.35, phenotype - preds))
        else:
            preds = np.clip(preds, a_min=1e-6, a_max=1. - 1e-6)
            return np.mean(-(self.class_weights[1] * phenotype * np.log(preds) +
                             self.class_weights[0] * (1. - phenotype) * np.log(1. - preds)))

    def objective(self):
        """
        The optimization objective
        """
        return self.complete_nll()

    def complete_nll(self):
        """
        The complete-data negative log-likelihood
        """

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

        if self.loss == 'mse':
            ll = -0.5 * (np.log(2. * np.pi) + self.log_resid +
                         np.exp(-self.log_resid) * self._expert_loss)
        else:
            ll = -self._expert_loss

        if axis is not None:
            return ll.sum(axis=axis)
        else:
            return ll

    def update_expert_losses(self):
        """
        Update the losses for each expert
        """

        preds = self.get_predictions()

        if self.loss == 'mse':
            self._expert_loss = (self.phenotype - preds) ** 2
        elif self.loss == 'huber':
            self._expert_loss = huber(1.35, (self.phenotype - preds))
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
            expert_predictions = self.expert_predictions.copy()
        else:
            expert_predictions = prs_dataset.get_data_columns(self.expert_cols, scaler=self.data_scaler)

        assert expert_predictions is not None

        # Process the expert-specific covariates (allowed to be None):
        if prs_dataset is None:
            expert_covariates = self.expert_covariates
        else:
            if self.expert_covariates_cols is not None:
                expert_covariates = prs_dataset.get_data_columns(self.expert_covariates_cols,
                                                                 add_intercept=self.expert_add_intercept,
                                                                 scaler=self.data_scaler)
            elif self.expert_add_intercept:
                expert_covariates = np.ones((prs_dataset.N, 1))
            else:
                expert_covariates = None

        # Step 1: Separate the last parameter and the remaining parameters in expert_params
        scale_params = self.expert_params[:, -1:]  # Shape: Kx1

        # Step 2: Multiply predictions by scaling parameter:
        predictions_mult = expert_predictions * scale_params.T  # Shape: NxK

        # Step 3: If expert-specific covariates are defined, use them to compute the "shift term":
        if expert_covariates is not None:
            shift_params = self.expert_params[:, :-1]  # Shape: KxC
            shift_term = expert_covariates.dot(shift_params.T)  # Shape: NxK
        else:
            shift_term = 0.

        # Step 4: Add the results of step 2 and step 3
        expert_predictions = shift_term + predictions_mult  # Shape: NxK

        return expert_predictions

    def get_predictions(self, prs_dataset=None):
        """
        Get the predictions of each expert
        """
        expert_predictions = self.get_scaled_predictions(prs_dataset)

        if self.global_params is not None:
            if prs_dataset is None:
                global_covariates = self.global_covariates
            else:
                global_covariates = prs_dataset.get_data_columns(self.global_covariates_cols,
                                                                 add_intercept=not self.expert_add_intercept,
                                                                 scaler=self.data_scaler)

            expert_predictions += global_covariates.dot(self.global_params.T).reshape(-1, 1)

        if self.loss == 'bce':
            return expit(expert_predictions)
        else:
            return expert_predictions

    def e_step(self):
        """
        The Expectation step of the EM algorithm. This function computes
        the log of the expert responsibilities.
        """
        self.log_resp = log_softmax(self.log_w + self.ll(), axis=-1)

    def m_step_decoupled(self):
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
                        a_min=-20., a_max=20.)
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

                reshaped_gparams = gparams.reshape(self.gate_dim, self.K - 1)

                log_g = log_softmax(
                    _concat_zero(gate_input.dot(reshaped_gparams)),
                    axis=-1
                )

                # Scale the loss by 1/sample_size for numerical stability?
                loss = - (1. / self.N) * np.sum(selected_expert_resp * log_g)
                grad = (1. / self.N) * gate_input.T.dot(np.exp(log_g[:, :-1]) - selected_expert_resp[:, :-1]).flatten()

                if self.gate_penalty > 0:

                    if self.gate_add_intercept:
                        # Remove the intercept term from the penalty:
                        reshaped_gparams[0, :] = 0.

                    loss += self.gate_penalty * (reshaped_gparams ** 2).sum()
                    grad += 2. * self.gate_penalty * reshaped_gparams.flatten()

                return loss, grad

            res = minimize(local_objective,
                           self.gate_params.flatten(),
                           jac=True,
                           method='L-BFGS-B')
            if not res.success:
                print("Gate parameter optimization not successful:\n", res)

            self.gate_params = res.x.reshape(self.gate_dim, self.K - 1)

        # -------------------------------------------------------------------------
        # (2) Update the parameters of the global covariates (if present):

        for _ in range(50):

            if self.global_covariates is not None:

                def fit_gaussian_model(w, y, S, C):

                    N, M = C.shape
                    # Compute weighted mean of S per row
                    S_bar = np.sum(w * S, axis=1)
                    r = y - S_bar  # target residuals

                    # Solve least squares
                    model = copy.deepcopy(self.wl_model)
                    model.fit_intercept = False
                    model.fit(C, r)

                    return model.coef_.flatten()

                self.global_params = fit_gaussian_model(expert_resp,
                                                        self.phenotype.flatten(),
                                                        self.get_scaled_predictions(),
                                                        self.global_covariates)

            # -------------------------------------------------------------------------
            # (3) Update the parameters of the experts:

            if self.expert_params is not None:

                if self.global_covariates is not None:
                    target = self.phenotype.flatten() - self.global_covariates.dot(self.global_params)
                else:
                    target = self.phenotype.flatten()

                def fit_model(i):
                    model = copy.deepcopy(self.wl_model)
                    if self.expert_covariates is not None:
                        x = np.concatenate([self.expert_covariates,
                                            self.expert_predictions[:, i, None]], axis=1)
                    else:
                        x = self.expert_predictions[:, i, None]

                    model.fit(x, target, sample_weight=expert_resp[:, i])

                    return model.coef_.flatten()

                self.expert_params = np.array(
                    Parallel(n_jobs=self.n_jobs)(
                        delayed(fit_model)(i)
                        for i in range(len(self.expert_params))
                    )
                )

        # -------------------------------------------------------------------------

        # (4) Given the updated parameters, re-compute the expert weights and losses:
        self.log_w = self.predict_proba(log=True)
        self.update_expert_losses()

        # -------------------------------------------------------------------------
        # (5) Update the residuals (for Gaussian likelihoods):
        self.update_residuals()

    def m_step(self, use_jac=True):
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
                        a_min=-20., a_max=20.)
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

                reshaped_gparams = gparams.reshape(self.gate_dim, self.K - 1)

                log_g = log_softmax(
                    _concat_zero(gate_input.dot(reshaped_gparams)),
                    axis=-1
                )

                # Scale the loss by 1/sample_size for numerical stability?
                loss = - (1. / self.N) * np.sum(selected_expert_resp * log_g)
                grad = (1. / self.N) * gate_input.T.dot(np.exp(log_g[:, :-1]) - selected_expert_resp[:, :-1]).flatten()

                if self.gate_penalty > 0:

                    if self.gate_add_intercept:
                        # Remove the intercept term from the penalty:
                        reshaped_gparams[0, :] = 0.

                    loss += self.gate_penalty * (reshaped_gparams ** 2).sum()
                    grad += 2. * self.gate_penalty * reshaped_gparams.flatten()

                return loss, grad

            res = minimize(local_objective,
                           self.gate_params.flatten(),
                           jac=True,
                           method='L-BFGS-B')
            if not res.success:
                print("Gate parameter optimization not successful:\n", res)

            self.gate_params = res.x.reshape(self.gate_dim, self.K - 1)

        # -------------------------------------------------------------------------

        def unpack_expert_params(params):
            param_start = 0

            if self.global_params is not None:
                self.global_params = params[param_start:self.global_params.size]
                param_start += self.global_params.size

            if self.expert_params is not None:
                self.expert_params = params[param_start:param_start + self.expert_params.size].reshape(
                    self.expert_params.shape
                )

            self.update_expert_losses()

        def pack_expert_params():

            init_params = []

            if self.global_params is not None:
                init_params.append(self.global_params.flatten())

            if self.expert_params is not None:
                init_params.append(self.expert_params.flatten())

            return np.concatenate(init_params)

        def joint_expert_objective(params):

            unpack_expert_params(params)

            if use_jac:
                return self.weighted_nll(), self._weighted_nll_grad()
            else:
                return self.weighted_nll()

        init_params = pack_expert_params()

        if len(init_params) > 0:
            res = minimize(joint_expert_objective,
                           init_params,
                           jac=use_jac,
                           method='L-BFGS-B',
                           options={"maxiter": 1000, "gtol": 1e-5, "ftol": 1e-8, })

            if not res.success:
                print("Expert parameter optimization not successful:\n", res)

            unpack_expert_params(res.x)

        # -------------------------------------------------------------------------
        # (4) Given the updated parameters, re-compute the expert weights and losses:
        self.log_w = self.predict_proba(log=True)
        self.update_expert_losses()

        # -------------------------------------------------------------------------
        # (5) Update the residuals (for Gaussian likelihoods):
        self.update_residuals()


    def update_residuals(self):
        """
        Update the residual variance for the experts (applicable when
        the phenotype likelihood is Gaussian).
        """

        if not self.fix_residuals and self.loss == 'mse':
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

        if prs_dataset is None:
            gate_input = self.gate_input
        elif self.gate_input_cols is not None:
            gate_input = prs_dataset.get_data_columns(self.gate_input_cols,
                                                      add_intercept=self.gate_add_intercept,
                                                      scaler=self.data_scaler)
        elif self.gate_add_intercept:
            gate_input = np.ones((prs_dataset.N, 1))
        else:
            raise ValueError("Gate model is not setup properly!")

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

        scaled_pred = self.get_predictions(prs_dataset)

        return (self.predict_proba(prs_dataset) * scaled_pred).sum(axis=1)

    def predict_prs(self, prs_dataset=None):
        """
        Predict the PRS for new samples (ignoring global covariates, if they are in the model).
        :param prs_dataset: An instance of `PRSDataset` containing the covariates to use as input
        for the gating model.
        """

        scaled_preds = self.get_scaled_predictions(prs_dataset)

        return (self.predict_proba(prs_dataset) * scaled_preds).sum(axis=1)

    def fit(self,
            continued=False,
            n_iter=1000,
            n_restarts=1,
            param_0=None,
            patience=10,
            atol=1e-5,
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
        if not continued:
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
                if not continued:
                    self.initialize(param_0=param_0, init_history=False)

            self.e_step()
            self.m_step()

            self.history['NCLL'].append(self.complete_nll())
            self.history['Weighted Loss'].append(self.weighted_loss())
            self.history['Weighted NLL'].append(self.weighted_nll())
            self.history['Gate Loss'].append(self.gate_loss())
            self.history['Ensemble Loss'].append(self.ensemble_loss())
            self.history['Expert Losses'].append(self._expert_loss.mean(axis=0))
            self.history['Model Weights'].append(self.expert_responsibility.mean(axis=0))

            #if curr_iter > 0 and curr_iter % 20 == 0:
            #    print(curr_iter)
            #    for k, v in self.get_model_parameters().items():
            #        print(k)
            #        print((v- self.param_tracker.current_params[k]) / self.param_tracker.current_params[k])

            if curr_iter > 0:
                if np.allclose(self.history[objective][-1], self.history[objective][-2], atol=atol, rtol=0.):
                    print(f"Objective converged at iteration {curr_iter}")
                    restart = True
                elif self.param_tracker.all_close(self.get_model_parameters(), atol=atol):
                    print(f"Parameters converged at iteration {curr_iter}")
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
            self.log_resid = self.param_tracker.best_params['log_resid'].values.copy().flatten()

        return self

    def get_model_parameters(self):

        gate_param_names = [[], ['Intercept']][self.gate_add_intercept]
        if self.gate_input_cols is not None:
            gate_param_names += self.gate_input_cols

        params = {
            'gate_params': pd.DataFrame(self.gate_params,
                                        index=gate_param_names,
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

        if self.global_params is not None:
            params['global_params'] = pd.DataFrame(self.global_params,
                                                   columns=['Coefficient'],
                                                   index=[[], ['Intercept']][not self.expert_add_intercept] + self.global_covariates_cols)

        if self.log_resid is not None:
            params['log_resid'] = pd.DataFrame(self.log_resid,
                                               columns=['Log Residual Variance'],
                                               index=self.expert_cols)

        return params

    def _params_to_init_dict(self):

        params = self.get_model_parameters()

        for key, val in params.items():
            params[key] = val.values

            if key == 'global_params':
                params[key] = params[key].flatten()

        return params

    def two_step_fit(self, **fit_kwargs):
        """
        Perform model fitting in two steps.
        """

        # First stage: Fit with
        curr_gate_input = self.gate_input.copy()
        gate_cols = copy.copy(self.gate_input_cols)

        self.gate_input_cols = None
        self.gate_input = np.ones((self.N, 1))

        self.fit(**fit_kwargs)

        params = self.get_model_parameters()

        for key, val in params.items():
            params[key] = val.values

            if key == 'global_params':
                params[key] = params[key].flatten()
            elif key == 'gate_params':
                params[key] = np.zeros((curr_gate_input.shape[1], self.K - 1))
                params[key][0, :] = val.values

        print(params)
        # Second stage: reset the gating model input and refit:
        self.gate_input_cols = gate_cols
        self.gate_input = curr_gate_input

        self.fit(param_0=params, **fit_kwargs)

        print(self.get_model_parameters())
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
                self.global_params,
                self.log_resid,
                self.gate_add_intercept,
                self.expert_add_intercept,
                self.gate_input_cols,
                self.expert_cols,
                self.expert_covariates_cols,
                self.global_covariates_cols,
                self.loss,
                self.data_scaler
            ], outf)
