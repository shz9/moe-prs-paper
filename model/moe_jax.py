import numpy as np
from jax import grad, jit, value_and_grad, numpy as jnp
from tqdm import tqdm
import jaxopt
from jax.nn import log_softmax, softmax, relu
from jax import random
from jax.scipy.special import logsumexp
from jax.example_libraries.optimizers import adam


def optimize(params_init, loss_fn, num_steps, step_size=1e-1):
    # Initialize optimizer.
    opt_init, opt_update, get_params = adam(step_size=step_size)
    opt_state = opt_init(params_init)

    # Initialize best params and loss.
    best_params = params_init
    best_loss = float('inf')

    # Define a training step.
    def step(i, opt_state):
        params = get_params(opt_state)
        value, grads = value_and_grad(loss_fn)(params)
        opt_state = opt_update(i, grads, opt_state)
        return value, opt_state

    # Run the training loop.
    for i in tqdm(range(num_steps), total=num_steps):
        loss, opt_state = step(i, opt_state)
        if loss < best_loss:
            best_loss = loss
            best_params = get_params(opt_state)

    return best_params

def _add_intercept(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return jnp.concatenate([jnp.ones((x.shape[0], 1)), x], axis=1)


def _concat_zero(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return jnp.concatenate([x, jnp.zeros((x.shape[0], 1))], axis=1)


class GateModel(object):

    def __init__(self, layer_sizes, add_bias=True, add_bias_first_layer=False):
        self.layer_sizes = layer_sizes
        self.add_bias = add_bias
        self.add_bias_first_layer = add_bias_first_layer
        self.params = self.initialize()

    def initialize(self):

        key = random.PRNGKey(0)
        params = []
        for i in range(len(self.layer_sizes) - 1):
            w_key, b_key = random.split(key)

            W = random.normal(w_key, (self.layer_sizes[i], self.layer_sizes[i+1]))

            if not self.add_bias or (i == 0 and not self.add_bias_first_layer):
                params.append((W, ))
            else:
                b = jnp.zeros(self.layer_sizes[i + 1])
                params.append((W, b))

        return params

    def get_flattened_params(self):
        weights = jnp.concatenate([p[0].flatten() for p in self.params])
        biases = [p[1].flatten() for p in self.params if len(p) > 1]

        if len(biases) > 0:
            biases = jnp.concatenate(biases)
            return jnp.concatenate([weights, biases])
        else:
            return weights

    def forward(self, inputs):
        activations = inputs
        for p in self.params[:-1]:

            outputs = jnp.dot(activations, p[0])

            if len(p) > 1:
                activations = relu(outputs + p[1])
            else:
                activations = relu(outputs)

        if len(self.params[-1]) > 1:
            output = jnp.dot(activations, self.params[-1][0]) + self.params[-1][1]
        else:
            output = jnp.dot(activations, self.params[-1][0])

        return output


class JaxLinearMoE(object):

    def __init__(self,
                 covariates,
                 expert_predictions,
                 phenotype,
                 loss='likelihood_mixture',
                 add_intercept=False,
                 gate_model_layers=None,
                 expert_penalty=0.,
                 gate_penalty=0.,
                 mode=None,
                 optimizer='L-BFGS-B'):

        assert loss in ('likelihood_mixture', 'ensemble_mixture')

        # Sanity checks:
        assert phenotype.shape[0] == covariates.shape[0] == expert_predictions.shape[0]
        assert mode in ('ssp', 'covariates', None)

        self.covariates = jnp.array(covariates)
        self.expert_predictions = jnp.array(expert_predictions)
        self.phenotype = jnp.array(phenotype.reshape(-1, 1))

        if add_intercept:
            self.covariates = _add_intercept(covariates)
        else:
            self.covariates = covariates

        self.mode = mode

        # ------------------------------------------
        # Initialize the gating model:
        if gate_model_layers is None:
            gate_model_layers = (self.C, self.K)
        else:
            gate_model_layers = (self.C, *gate_model_layers, self.K)

        self.gate_model = GateModel(gate_model_layers)
        # ------------------------------------------

        self.expert_params = None
        self.add_intercept = add_intercept

        self.gate_penalty = gate_penalty
        self.expert_penalty = expert_penalty
        self.optimizer = optimizer

        self.loss_func = loss

        self.loss_history = []

    @property
    def N(self):
        """
        The number of samples
        """
        return self.covariates.shape[0]

    @property
    def C(self):
        """
        The number of input covariates to the gating model
        """
        return self.covariates.shape[1]

    @property
    def K(self):
        """
        The number of experts
        """
        return self.expert_predictions.shape[1]

    def predict_proba(self, covar=None, log=False):
        """
        Predict the expert weights based on an input set of covariates.
        :param covar: The covariates to use as input for the gating model
        (default: None; Use training covariates)
        :param log: If True, return log probabilities.
        """

        if covar is None:
            covar = self.covariates
        else:
            if self.add_intercept or not self.gate_model.add_bias_first_layer:
                covar = _add_intercept(covar)

        logits = self.gate_model.forward(covar)

        if log:
            return log_softmax(logits, axis=-1)
        else:
            return softmax(logits, axis=-1)

    def get_scaled_predictions(self, covar=None, predictions=None):
        """
        This function applies affine transformations (linear scaling and shifting) of
         expert predictions based on the current value of `self.expert_params`.
        """

        if predictions is None:
            predictions = self.expert_predictions

        if self.mode is None:
            return predictions
        else:

            if self.mode == 'ssp':
                covar = jnp.ones((predictions.shape[0], 1))
            else:
                if covar is None:
                    covar = self.covariates
                else:
                    if self.add_intercept:
                        covar = _add_intercept(covar)

            # Step 1: Separate the last parameter and the remaining parameters in expert_params
            scale_params = self.expert_params[:, -1:]  # Shape: Kx1
            shift_params = self.expert_params[:, :-1]  # Shape: KxC

            # Step 2: Calculate the dot product of covariates and sp_remaining
            dot_product = covar.dot(shift_params.T)  # Shape: NxK

            # Step 3: Multiply predictions by sp_last
            predictions_mult = predictions * scale_params.T  # Shape: NxK

            # Step 4: Add the results of step 2 and step 3
            return dot_product + predictions_mult  # Shape: NxK

    def predict(self, covar, predictions, weighted_only=False, *args, **kwargs):
        """
        Predict for new samples
        """

        if weighted_only or self.expert_params is None:

            return (
                    self.predict_proba(covar) * predictions
            ).sum(axis=1)

        else:
            return (
                    self.predict_proba(covar) * self.get_scaled_predictions(
                covar, predictions
            )).sum(axis=1)

    def loss(self, params, covar=None, predictions=None, phenotype=None):

        self.gate_model.params, self.expert_params = params

        if predictions is not None:
            assert covar is not None and phenotype is not None

        weights = self.predict_proba(covar)
        scaled_preds = self.get_scaled_predictions(covar, predictions)

        if phenotype is None:
            phenotype = self.phenotype

        n = phenotype.shape[0]

        penalty = 0.

        if self.gate_penalty > 0:
            penalty = self.gate_penalty*np.sum(self.gate_model.get_flattened_params()**2)

        if self.expert_params is not None and self.expert_penalty > 0.:
            penalty += self.expert_penalty*(self.expert_params**2).sum()

        if self.loss_func == 'likelihood_mixture':
            squared_error = (phenotype - scaled_preds) ** 2
            loss = (1./n)*(weights * squared_error).sum() + penalty
        elif self.loss_func == 'ensemble_mixture':
            weighted_preds = (weights * scaled_preds).sum(axis=1, keepdims=True)
            loss = (1./n)*((phenotype - weighted_preds) ** 2).sum() + penalty

        try:
            self.loss_history.append(np.asarray(loss.primal))
        except Exception as e:
            pass

        return loss

    def fit(self, n_iter=1000):

        self.loss_history = []

        self.gate_model.initialize()

        if self.mode == 'ssp':
            expert_params = jnp.array(np.random.normal(size=(self.K, 2)))
            params = (self.gate_model.params, expert_params)
        elif self.mode == 'covariates':
            expert_params = jnp.array(np.random.normal(size=(self.K, self.C + 1)))
            params = (self.gate_model.params, expert_params)
        else:
            params = (self.gate_model.params, None)

        if self.optimizer == 'L-BFGS-B':
            solver = jaxopt.LBFGS(fun=self.loss, maxiter=n_iter, tol=1e-6)
            res = solver.run(params)

            self.gate_model.params, self.expert_params = res.params
        else:
            self.gate_model.params, self.expert_params = optimize(params, self.loss, n_iter)

        return self
