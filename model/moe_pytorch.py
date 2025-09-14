import pickle
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
import lightning.pytorch as pl

try:
    from sparsemax import Sparsemax
except ImportError:
    pass

import argparse
from PRSDataset import PRSDataset


class ConvergenceCheck(pl.callbacks.Callback):

    def __init__(self, rtol=1e-05, atol=1e-05):

        super().__init__()

        self.rtol = rtol
        self.atol = atol

        self.best_loss = np.inf
        self.best_params = None

    def on_train_start(self, trainer, pl_module):

        self.best_loss = np.inf
        self.best_params = None

    def on_train_end(self, trainer, pl_module):

        current_loss = trainer.callback_metrics['train_loss']
        current_params = [p.detach().numpy().copy() for p in pl_module.parameters()]

        if self.best_params is not None:
            if np.allclose(current_loss, self.best_loss, rtol=self.rtol, atol=self.atol):
                print("> Convergence achieved (negligible change in objective)")
                trainer.should_stop = True
            elif all([np.allclose(p1, p2, rtol=self.rtol, atol=self.atol)
                      for p1, p2 in zip(current_params, self.best_params)]):
                print("> Convergence achieved (negligible change in parameters)")
                trainer.should_stop = True

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_params = current_params
        else:
            self.best_loss = current_loss
            self.best_params = current_params


#########################################################


def likelihood_mixture_loss(expert_weights, expert_predictions, phenotype, family="gaussian"):

    assert family in ("gaussian", "binomial")

    N = expert_weights.shape[0]

    if family == "gaussian":
        losses = (expert_predictions - phenotype)**2
    else:
        expert_predictions = torch.clamp(expert_predictions, 1e-6, 1.-1e-6)
        losses = -(phenotype*torch.log(expert_predictions) + (1.-phenotype)*torch.log(1.-expert_predictions))

    return (1./N)*(expert_weights*losses).sum()


def likelihood_mixture_loss2(expert_weights, expert_predictions, phenotype, family="gaussian"):
    """
    An alternative loss that matches the likelihood mixture loss outlined in Equation (1.3)
    in Jacobs et al. 1991.
    """

    assert family in ("gaussian", "binomial")

    if family == "gaussian":
        lik = torch.exp(-.5*(expert_predictions - phenotype)**2)
    else:
        expert_predictions = torch.clamp(expert_predictions, 1e-6, 1.-1e-6)
        lik = torch.exp(phenotype*torch.log(expert_predictions) + (1.-phenotype)*torch.log(1.-expert_predictions))

    return -torch.log((expert_weights*lik).sum(axis=1)).mean()


def ensemble_mixture_loss(expert_weights, expert_predictions, phenotype, family="gaussian"):

    assert family in ("gaussian", "binomial")

    pred = (expert_weights*expert_predictions).sum(axis=1)

    if family == "gaussian":
        return ((pred - phenotype)**2).mean()
    else:
        pred = torch.clamp(pred, 1e-6, 1.-1e-6)
        return -(phenotype*torch.log(pred) + (1.-phenotype)*torch.log(1.-pred)).mean()


def ensemble_mixture_loss_simple(phenotype, pred, family="gaussian"):

    assert family in ("gaussian", "binomial")

    if family == "gaussian":
        return ((pred - phenotype)**2).mean()
    else:
        pred = torch.clamp(pred, 1e-6, 1.-1e-6)
        return -(phenotype*torch.log(pred) + (1.-phenotype)*torch.log(1.-pred)).mean()

#########################################################
# Define a PyTorch Lightning module to streamline training

class Lit_MoEPRS(pl.LightningModule):

    def __init__(self,
                 group_getitem_cols,
                 gate_model_layers=None,
                 gate_add_batch_norm=True,
                 loss="likelihood_mixture",
                 optimizer="Adam",
                 family="gaussian",
                 learning_rate=1e-3,
                 weight_decay=0.):
        """
        A PyTorch Lightning module for training a mixture of experts model.

        :param group_getitem_cols: A dictionary mapping categories of data to the relevant keys from the
         pandas dataframe. This is useful for iterative data fetching (e.g. data loaders).
            These are used to define what columns/groups of columns are fetched in the __getitem__ method.
        :param gate_model_layers: A list of integers specifying the number of hidden units
        in the gating model.
        :param gate_add_batch_norm: If True, add batch normalization to the gating model.
        :param loss: The loss function to use. Options are: ('likelihood_mixture', 'ensemble_mixture')
        :param optimizer: The optimizer to use. Options are: ('Adam', 'LBFGS', 'SGD')
        :param family: The family of the likelihood. Options are: ('gaussian', 'binomial')
        :param learning_rate: The learning rate for the optimizer.
        :param weight_decay: The weight decay for the optimizer.
        """

        super().__init__()

        # -------------------------------------------------------
        # Sanity checks for the inputs:
        assert loss in ("likelihood_mixture",
                        "likelihood_mixture2",
                        "ensemble_mixture")
        assert optimizer in ("Adam", "LBFGS", "SGD")
        assert family in ("gaussian", "binomial")

        assert 'phenotype' in group_getitem_cols
        assert 'gate_input' in group_getitem_cols
        assert 'experts' in group_getitem_cols

        # -------------------------------------------------------
        # Define / initialize the model components:

        self.group_getitem_cols = group_getitem_cols

        self.gate_model = GateModel(self.gate_input_dim,
                                    self.n_experts,
                                    hidden_layers=gate_model_layers,
                                    add_batch_norm=gate_add_batch_norm)  # The gating model

        if self.n_expert_covariates > 0:
            self.expert_scaler = nn.ModuleList([LinearScaler(self.n_expert_covariates, family=family)
                                                for _ in range(self.n_experts)])
        else:
            self.expert_scaler = nn.ModuleList([LinearScaler(family=family)
                                                for _ in range(self.n_experts)])

        self.loss = loss
        self.metrics = {
            'likelihood_mixture': partial(likelihood_mixture_loss, family=family),
            'ensemble_mixture': partial(ensemble_mixture_loss, family=family),
            'likelihood_mixture2': partial(likelihood_mixture_loss2, family=family)
        }

        # Optimizer options:
        self.family = family
        self.optimizer = optimizer
        self.lr = learning_rate
        self.weight_decay = weight_decay

    @property
    def n_experts(self):
        return len(self.group_getitem_cols['experts'])

    @property
    def gate_input_dim(self):
        return len(self.group_getitem_cols['gate_input'])

    @property
    def n_expert_covariates(self):
        if 'expert_covariates' in self.group_getitem_cols:
            return len(self.group_getitem_cols['expert_covariates'])
        else:
            return 0

    def batch_step(self, batch, batch_idx):

        proba = self.gate_forward(batch)
        scaled_pred = self.scale_expert_predictions(batch)

        losses = {}

        for m, loss in self.metrics.items():
            losses[m] = loss(proba, scaled_pred, batch['phenotype'])

            # If we're using L-BFGS for optimization, add weight decay manually:
            if self.weight_decay > 0. and self.optimizer == "LBFGS":
                losses[m] += self.weight_decay * torch.norm(self.gate_model.gate[0].weight, p=2)
                if self.n_expert_covariates > 0:
                    for expert in self.expert_scaler:
                        losses[m] += self.weight_decay * torch.norm(expert.linear_model.weight, p=2)

        return losses

    def training_step(self, batch, batch_idx):

        losses = self.batch_step(batch, batch_idx)

        self.log("train_loss", losses[self.loss], prog_bar=True)

        for m, loss in losses.items():
            if m != self.loss:
                self.log(m, loss, prog_bar=True)

        return losses[self.loss]

    def validation_step(self, batch, batch_idx):
        losses = self.batch_step(batch, batch_idx)
        self.log("val_loss", losses[self.loss], prog_bar=True)

        return losses[self.loss]

    def scale_expert_predictions(self, batch):

        if 'expert_covariates' in batch:
            expert_covariates = batch['expert_covariates']
        else:
            expert_covariates = None

        return torch.cat([expert_scaler.forward(batch['experts'][:, i], covar=expert_covariates)
                          for i, expert_scaler in enumerate(self.expert_scaler)],
                         dim=1)

    def gate_forward(self, batch):
        return self.gate_model.forward(batch['gate_input'])

    def forward(self, batch):
        return (self.gate_forward(batch)*self.scale_expert_predictions(batch)).sum(axis=1)

    def predict(self, batch):

        if isinstance(batch, dict):
            return self.forward(batch)
        else:
            return self.predict_from_dataset(batch)

    def predict_from_dataset(self, prs_dataset):

        # Sanity checks:
        assert 'experts' in prs_dataset.group_getitem_cols
        assert 'gate_input' in prs_dataset.group_getitem_cols
        assert self.group_getitem_cols['experts'] == prs_dataset.group_getitem_cols['experts']
        assert self.group_getitem_cols['gate_input'] == prs_dataset.group_getitem_cols['gate_input']

        prs_dataset.set_backend("torch")

        dat = DataLoader(prs_dataset, batch_size=prs_dataset.N, shuffle=False)

        return self.forward(next(iter(dat))).detach().numpy()

    def predict_proba(self, batch):

        if isinstance(batch, dict):
            return self.gate_model(batch)
        else:
            return self.predict_proba_from_dataset(batch)

    def predict_proba_from_dataset(self, prs_dataset):

        assert 'gate_input' in prs_dataset.group_getitem_cols
        assert self.group_getitem_cols['gate_input'] == prs_dataset.group_getitem_cols['gate_input']

        prs_dataset.set_backend("torch")

        dat = DataLoader(prs_dataset, batch_size=prs_dataset.N, shuffle=False)

        return self.gate_forward(next(iter(dat))).detach().numpy()

    def configure_optimizers(self):

        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.LBFGS(self.parameters())

        return optimizer


class Lit_MoEPRS2(pl.LightningModule):

    def __init__(self,
                 group_getitem_cols,
                 optimizer="Adam",
                 family="gaussian",
                 learning_rate=1e-3,
                 weight_decay=0.):
        """
        A PyTorch Lightning module for training a mixture of experts model.

        :param group_getitem_cols: A dictionary mapping categories of data to the relevant keys from the
         pandas dataframe. This is useful for iterative data fetching (e.g. data loaders).
            These are used to define what columns/groups of columns are fetched in the __getitem__ method.
        :param gate_model_layers: A list of integers specifying the number of hidden units
        in the gating model.
        :param gate_add_batch_norm: If True, add batch normalization to the gating model.
        :param loss: The loss function to use. Options are: ('likelihood_mixture', 'ensemble_mixture')
        :param optimizer: The optimizer to use. Options are: ('Adam', 'LBFGS', 'SGD')
        :param family: The family of the likelihood. Options are: ('gaussian', 'binomial')
        :param learning_rate: The learning rate for the optimizer.
        :param weight_decay: The weight decay for the optimizer.
        """

        super().__init__()

        # -------------------------------------------------------
        # Sanity checks for the inputs:
        assert optimizer in ("Adam", "LBFGS", "SGD")
        assert family in ("gaussian", "binomial")

        assert 'phenotype' in group_getitem_cols
        assert 'gate_input' in group_getitem_cols
        assert 'experts' in group_getitem_cols

        # -------------------------------------------------------
        # Define / initialize the model components:

        self.group_getitem_cols = group_getitem_cols

        self.model = MoEWithLinear(self.gate_input_dim,
                                   self.n_experts,
                                   family=family)

        self.loss = partial(ensemble_mixture_loss_simple, family=family)

        # Optimizer options:
        self.family = family
        self.optimizer = optimizer
        self.lr = learning_rate
        self.weight_decay = weight_decay

    @property
    def n_experts(self):
        return len(self.group_getitem_cols['experts'])

    @property
    def gate_input_dim(self):
        return len(self.group_getitem_cols['gate_input'])

    def batch_step(self, batch, batch_idx):

        loss = self.loss(batch['phenotype'], self.forward(batch), batch['phenotype'])

        # If we're using L-BFGS for optimization, add weight decay manually:
        if self.weight_decay > 0. and self.optimizer == "LBFGS":
            loss += self.weight_decay * torch.norm(self.gate_model.gate[0].weight, p=2)
            if self.n_expert_covariates > 0:
                for expert in self.expert_scaler:
                    loss += self.weight_decay * torch.norm(expert.linear_model.weight, p=2)

        return loss

    def training_step(self, batch, batch_idx):

        loss = self.batch_step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.batch_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def gate_forward(self, batch):
        return self.model.gating.forward(batch['gate_input'])

    def forward(self, batch):
        return self.model.forward(batch['gate_input'], batch['experts'])

    def predict(self, batch):
        return self.forward(batch)

    def predict_from_dataset(self, prs_dataset):

        # Sanity checks:
        assert 'experts' in prs_dataset.group_getitem_cols
        assert 'gate_input' in prs_dataset.group_getitem_cols
        assert self.group_getitem_cols['experts'] == prs_dataset.group_getitem_cols['experts']
        assert self.group_getitem_cols['gate_input'] == prs_dataset.group_getitem_cols['gate_input']

        prs_dataset.set_backend("torch")

        dat = DataLoader(prs_dataset, batch_size=prs_dataset.N, shuffle=False)

        return self.predict(next(iter(dat))).detach().numpy()

    def predict_proba(self, batch):
        return self.gate_forward(batch)

    def predict_proba_from_dataset(self, prs_dataset):

        assert 'gate_input' in prs_dataset.group_getitem_cols
        assert self.group_getitem_cols['gate_input'] == prs_dataset.group_getitem_cols['gate_input']

        prs_dataset.set_backend("torch")

        dat = DataLoader(prs_dataset, batch_size=prs_dataset.N, shuffle=False)

        return self.predict_proba(next(iter(dat))).detach().numpy()

    def configure_optimizers(self):

        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.LBFGS(self.parameters())

        return optimizer

#########################################################

class MoEWithLinear(nn.Module):
    def __init__(self, num_covariates, num_experts, family='gaussian'):

        super().__init__()

        assert family in ('gaussian', 'binomial')

        self.num_experts = num_experts
        self.family = family

        # Linear model over covariates (includes bias)
        self.linear = nn.Linear(num_covariates, 1)

        # One linear expert per x dimension (just weight + bias)
        self.expert_weights = nn.Parameter(torch.randn(num_experts))  # shape (m,)
        self.expert_biases = nn.Parameter(torch.randn(num_experts))   # shape (m,)

        # Gating model that uses C to produce weights over experts
        self.gating = GateModel(num_covariates, num_experts)

        if family == 'gaussian':
            self.final_activation = nn.Identity()
        else:
            self.final_activation = nn.Sigmoid()

    def forward(self, C, x):
        """
        Inputs:
        - C: (N, k) covariates
        - x: (N, m) input to experts
        Output:
        - y_pred: (N,) predicted response
        """
        N, m = x.shape
        assert m == self.num_experts, "Each expert should correspond to one x dimension"

        # Linear part from covariates
        linear_out = self.linear(C).squeeze(-1)  # shape (N,)

        # Expert outputs: apply weight and bias to each x[:, i]
        expert_outputs = x * self.expert_weights + self.expert_biases  # shape (N, m)

        # Gating weights from C
        gate_weights = self.gating(C)  # shape (N, m), softmax over experts

        # Weighted sum of expert outputs
        moe_out = torch.sum(gate_weights * expert_outputs, dim=1)  # shape (N,)

        # Final output
        return self.final_activation(linear_out + moe_out)


# Define the gating model:

class GateModel(nn.Module):
    """
    A generic implementation for the gating model. This function can accommodate
    linear + non-linear gating models.
    """

    def __init__(self,
                 n_covar,
                 n_experts,
                 hidden_layers=None,
                 add_batch_norm=True,
                 activation=nn.ReLU,
                 # GELU
                 final_activation="softmax"):

        super(GateModel, self).__init__()

        self.n_covar = n_covar
        self.n_experts = n_experts

        input_dim = n_covar  # The input dimension for the gating model
        layers = []

        if hidden_layers is not None:

            input_dim = n_covar
            for layer_dim in hidden_layers:
                if len(layers) < 1:

                    layers.append(nn.Linear(input_dim, layer_dim))

                    if add_batch_norm:
                        layers.append(nn.BatchNorm1d(layer_dim))

                    layers.append(activation())

                    input_dim = layer_dim

        # Add the final layer:
        layers.append(nn.Linear(input_dim, n_experts))
        # Add the softmax activation:
        if final_activation == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif final_activation == "sparsemax":
            layers.append(Sparsemax(dim=1))

        self.gate = nn.Sequential(*layers)

    def forward(self, covar):
        return self.gate(covar)

    def predict_proba(self, covar):
        return self.forward(covar)


class LinearScaler(nn.Module):
    """
    A linear model for scaling the predictions of the experts.
    """

    def __init__(self, n_covar=0, bias=True, family="gaussian"):

        super(LinearScaler, self).__init__()

        assert family in ("gaussian", "binomial")

        # The linear model takes as inputs 1 (for the PRS) + the number of covariates.
        # If there are no covariates, it just takes the PRS itself.
        self.linear_model = nn.Linear(n_covar + 1, 1, bias=bias)
        self.family = family

    def forward(self, prs, covar=None):

        if len(prs.shape) < 2:
            prs = prs.reshape(-1, 1)

        if covar is None:
            pred = self.linear_model(prs)
        else:
            pred = self.linear_model(torch.cat([prs, covar], dim=1))

        if self.family == "gaussian":
            return pred
        else:
            return torch.sigmoid(pred)


def get_weighted_batch_sampler(dataset):

    try:
        targets = dataset.get_phenotype()
    except AttributeError:
        # If it's a subset of a dataset, extract the phenotype and then subset
        # for the given indices:
        targets = dataset.dataset.get_phenotype()[dataset.indices]

    # Compute samples weights
    class_sample_count = torch.tensor([(targets == t).sum() for t in [0, 1]])
    weight = 1. / class_sample_count.float()
    samples_weight = weight[targets.int()]

    # Create a weighted random sampler
    sampler = WeightedRandomSampler(samples_weight, targets.shape[0])

    return sampler

#########################################################


def train_model(lit_model, dataset, max_epochs=100, prop_validation=0.2, batch_size=None, weigh_samples=False):

    dataset.set_backend("torch")

    # Split the dataset into training and validation sets:

    if dataset.phenotype_likelihood == "binomial":
        stratify = dataset.get_phenotype()
    else:
        stratify = None

    dataset.standardize_data()

    train_idx, validation_idx = train_test_split(np.arange(dataset.N),
                                                 test_size=prop_validation,
                                                 shuffle=True,
                                                 stratify=stratify)

    training_dataset = Subset(dataset, train_idx)
    validation_dataset = Subset(dataset, validation_idx)

    if batch_size is not None:
        batch_size = min(batch_size, train_idx.shape[0], validation_idx.shape[0])

    if dataset.phenotype_likelihood == "binomial" and weigh_samples:
        train_sampler = get_weighted_batch_sampler(training_dataset)
        validation_sampler = get_weighted_batch_sampler(validation_dataset)
    else:
        train_sampler = None
        validation_sampler = None

    training_dataloader = DataLoader(training_dataset,
                                     batch_size=batch_size or train_idx.shape[0],
                                     shuffle=train_sampler is None,
                                     sampler=train_sampler)

    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=batch_size or validation_idx.shape[0],
                                       shuffle=validation_sampler is None,
                                       sampler=validation_sampler)

    ckpt_callback = pl.callbacks.ModelCheckpoint(
                                save_top_k=1,
                                monitor="val_loss",
                            )

    trainer = pl.Trainer(max_epochs=max_epochs,
                         callbacks=[
                            pl.callbacks.EarlyStopping(
                                monitor="val_loss",
                                patience=10,
                                check_finite=True,
                                check_on_train_epoch_end=True,
                                verbose=False
                            ),
                            ckpt_callback,
                            ConvergenceCheck()
                        ])

    trainer.fit(model=lit_model,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloader)

    ckpt = torch.load(ckpt_callback.best_model_path)
    lit_model.load_state_dict(ckpt['state_dict'])
    lit_model.eval()

    return trainer, lit_model


def main(args):

    model_root_dir = f"model/saved_models/{args.train_data}/PyTorch_models/"

    with open(f"data/harmonized_data/{args.train_data}.dat", "rb") as dfi:
        dataset = pickle.load(dfi)

    data_loader = DataLoader(dataset, batch_size=args.batch_size or len(dataset), shuffle=True)

    if args.gate == 'linear':
        gate_model = LinearGate(dataset.n_covariates,
                                dataset.n_experts,
                                final_activation=args.final_layer)
    else:
        gate_model = MLPGate(dataset.n_covariates,
                             dataset.n_experts,
                             final_activation=args.final_layer)

    if args.expert_scaler == "ssp":
        expert_scaler = LinearScale(dataset.n_experts)
    elif args.expert_scaler == "covariates":
        expert_scaler = LinearScale(dataset.n_experts, dataset.n_covariates)
    else:
        expert_scaler = None

    fname = f"{args.method}_{args.gate}_{args.final_layer}_lr{args.lr}_wd{args.weight_decay}"

    if expert_scaler is not None:
        fname += f"_scale{args.expert_scaler}"

    lit_model = Lit_GatingMoE(gate_model,
                              expert_scaler=expert_scaler,
                              method=args.method,
                              learning_rate=args.lr,
                              weight_decay=args.weight_decay)

    trainer = pl.Trainer(default_root_dir=model_root_dir,
                         log_every_n_steps=1,
                         max_epochs=args.max_epochs,
                         callbacks=[
                            pl.callbacks.EarlyStopping(
                                monitor="train_loss",
                                patience=10,
                                check_finite=True,
                                check_on_train_epoch_end=True,
                                verbose=True
                            ),
                            pl.callbacks.ModelCheckpoint(
                                filename=fname,
                                dirpath=model_root_dir,
                                save_top_k=1,
                                monitor="train_loss",
                            ),
                            ConvergenceCheck()
                        ])
    trainer.fit(model=lit_model, train_dataloaders=data_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train gating models impelemted in PyTorch")
    parser.add_argument("--training-data", dest="train_data", type=str, required=True)
    parser.add_argument("--final-layer", dest="final_layer", type=str, default="softmax")
    parser.add_argument("--method", dest="method", type=str, default="EM")
    parser.add_argument("--gate", dest="gate", type=str, default="linear",
                        choices={'linear', 'mlp'})
    parser.add_argument("--expert-scaler", dest="expert_scaler", type=str, default=None,
                        choices={'None', 'ssp', 'covariates'})
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.)
    parser.add_argument("--batch-size", dest="batch_size", type=str)
    parser.add_argument("--max-epochs", dest="max_epochs", type=int, default=1000)

    args = parser.parse_args()

    main(args)
