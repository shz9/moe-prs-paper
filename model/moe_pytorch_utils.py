import copy
import os

import lightning.pytorch as pl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def _as_list(x):
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def configure_cpu_threads(
    cpus_per_task: int, num_workers: int, interop_threads: int = 1
):
    """
    Give the main training process most of the CPU threads, leave some CPU capacity
    for DataLoader workers.
    """
    compute_threads = max(1, cpus_per_task - num_workers)

    # PyTorch CPU threading (main process)
    torch.set_num_threads(compute_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    # BLAS/threaded libs (main process)
    os.environ["OMP_NUM_THREADS"] = str(compute_threads)
    os.environ["MKL_NUM_THREADS"] = str(compute_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(compute_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(compute_threads)

    return compute_threads


def dataloader_worker_init_fn(worker_id: int):
    """
    Prevent each DataLoader worker from spawning its own CPU threadpool.
    This avoids massive oversubscription when num_workers > 0.
    """
    import os

    import torch

    torch.set_num_threads(1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def make_deterministic(seed: int = 16384):
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

    # PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


################################################################################
# Callbacks to determine stopping criteria:


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
        current_loss_t = trainer.callback_metrics["train_loss"]
        current_loss = (
            float(current_loss_t.detach().cpu().item())
            if isinstance(current_loss_t, torch.Tensor)
            else float(current_loss_t)
        )

        current_params = [p.detach().numpy().copy() for p in pl_module.parameters()]

        if self.best_params is not None:
            if np.allclose(
                current_loss, self.best_loss, rtol=self.rtol, atol=self.atol
            ):
                print("> Convergence achieved (negligible change in objective)")
                trainer.should_stop = True
            elif all(
                [
                    np.allclose(p1, p2, rtol=self.rtol, atol=self.atol)
                    for p1, p2 in zip(current_params, self.best_params)
                ]
            ):
                print("> Convergence achieved (negligible change in parameters)")
                trainer.should_stop = True

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_params = current_params
        else:
            self.best_loss = current_loss
            self.best_params = current_params


class DelayedEarlyStopping(pl.callbacks.Callback):
    def __init__(
        self, monitor="val_loss", min_epoch=100, patience=10, mode="min", min_delta=0.0
    ):
        super().__init__()
        self.monitor = monitor
        self.min_epoch = int(min_epoch)
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = None
        self.num_bad = 0

    def on_fit_start(self, trainer, pl_module):
        self.best = None
        self.num_bad = 0

    def on_validation_end(self, trainer, pl_module):
        # trainer.current_epoch is 0-indexed; compare using 1-indexed epoch count
        epoch = int(trainer.current_epoch) + 1
        if epoch < self.min_epoch:
            return

        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor, None)
        if current is None:
            return

        if isinstance(current, torch.Tensor):
            current = float(current.detach().cpu().item())
        else:
            current = float(current)

        if self.best is None:
            self.best = current
            self.num_bad = 0
            return

        if self.mode == "min":
            improved = current < (self.best - self.min_delta)
        else:
            improved = current > (self.best + self.min_delta)

        if improved:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                trainer.should_stop = True


################################################################################


def train_lit_model(
    lit_model,
    dataset,
    min_epochs=50,
    max_epochs=500,
    prop_validation=0.1,
    min_validation=1000,
    batch_size=None,
    weigh_samples=False,
    seed=8,
    ancestry_balance_lambda=0.3,
    standardize_data=True,
):
    dataset.set_backend("torch")

    # Split the dataset into training and validation sets:

    if dataset.phenotype_likelihood == "binomial":
        stratify = dataset.get_phenotype()
    else:
        stratify = None

    test_size = max(min_validation, int(prop_validation*dataset.N))

    # Split the dataset:
    train_idx, validation_idx = train_test_split(
        np.arange(dataset.N),
        test_size=test_size,
        shuffle=True,
        stratify=stratify,
        random_state=seed,
    )

    # Standardize the dataset:
    if standardize_data:
        dataset.standardize_data()

    # cache once AFTER standardization
    dataset.cache_data_matrix()

    training_dataset = IndexSubset(dataset, train_idx)
    validation_dataset = IndexSubset(dataset, validation_idx)

    if batch_size is not None:
        batch_size = min(batch_size, len(training_dataset), len(validation_dataset))

    # samplers (your existing helpers should still work because IndexSubset has .dataset and .indices)
    if dataset.phenotype_likelihood == "binomial" and weigh_samples:
        train_sampler = get_weighted_batch_sampler(training_dataset)
        validation_sampler = get_weighted_batch_sampler(validation_dataset)
    else:
        if ancestry_balance_lambda is not None:
            train_sampler = get_ancestry_balanced_sampler(
                training_dataset,
                balance_lambda=ancestry_balance_lambda,
            )
        else:
            train_sampler = None
        validation_sampler = None

    # IMPORTANT: collate_fn does the vectorized fetch
    collate = dataset.get_batch

    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

    # Good default for your cached+vectorized pipeline:
    train_workers = 3
    val_workers = 1
    prefetch_factor = 1

    configure_cpu_threads(
        cpus_per_task=cpus, num_workers=(train_workers + val_workers), interop_threads=1
    )

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size or len(training_dataset),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_workers,
        collate_fn=collate,
        worker_init_fn=dataloader_worker_init_fn,
        persistent_workers=(train_workers > 0),
        prefetch_factor=prefetch_factor,
        pin_memory=False,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size or len(validation_dataset),
        shuffle=False,
        sampler=validation_sampler,
        num_workers=val_workers,
        collate_fn=collate,
        worker_init_fn=dataloader_worker_init_fn,
        persistent_workers=(val_workers > 0),
        prefetch_factor=prefetch_factor,
        pin_memory=False,
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
    )

    # one directory per phenotype (sufficient if one job per pheno)
    # log_dir = osp.join("lightning_logs", dataset.phenotype_col)
    # logger = CSVLogger(save_dir=log_dir, name="MoE-PyTorch")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        deterministic=True,
        #  logger = logger,
        logger=False,
        num_sanity_val_steps=0,
        callbacks=[
            DelayedEarlyStopping(
                monitor="val_loss", min_epoch=min_epochs, patience=10, mode="min"
            ),
            ckpt_callback,
            ConvergenceCheck(),
        ],
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader,
    )

    ckpt = torch.load(ckpt_callback.best_model_path, weights_only=False)
    lit_model.load_state_dict(ckpt["state_dict"])
    lit_model.eval()
    lit_model.training_scaler = copy.deepcopy(dataset.scaler)

    return trainer, lit_model


################################################################################
# Data and batch-sampling related utils:


class IndexSubset(Dataset):
    """
    Like torch.utils.data.Subset, but returns the ORIGINAL row index (int),
    so collate_fn can fetch the whole batch via PRSDataset.get_batch().
    Keeps .dataset and .indices so your sampler helpers still work.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])


def get_weighted_batch_sampler(dataset):
    try:
        targets = dataset.get_phenotype()
    except AttributeError:
        # If it's a subset of a dataset, extract the phenotype and then subset
        # for the given indices:
        targets = dataset.dataset.get_phenotype()[dataset.indices]

    # Compute samples weights
    class_sample_count = torch.tensor([(targets == t).sum() for t in [0, 1]])
    weight = 1.0 / class_sample_count.float()
    samples_weight = weight[targets.int()]

    # Create a weighted random sampler
    sampler = WeightedRandomSampler(samples_weight, targets.shape[0])

    return sampler


def get_ancestry_balanced_sampler(dataset, balance_lambda: float = 0.3):
    """
    Weighted sampler that interpolates between:
      - empirical ancestry distribution p_data
      - uniform ancestry distribution

    balance_lambda = 0.0  -> no rebalancing (p_data)
    balance_lambda = 1.0  -> fully uniform over ancestries
    """

    # get ancestry
    try:
        ancestry = dataset.get_ancestry()  # e.g. np.array of strings or ints
    except AttributeError:
        # If we're passed a Subset, pull from the parent dataset and index
        ancestry = dataset.dataset.get_ancestry()[dataset.indices]

    # map if needed
    ancestries_unique, ancestry_ids = np.unique(ancestry, return_inverse=True)
    ancestry_tensor = torch.as_tensor(ancestry_ids, dtype=torch.long)  # (N,)

    # Counts and empirical distribution p_data
    class_sample_count = torch.bincount(ancestry_tensor)  # (K,)
    class_sample_count = class_sample_count.clamp_min(1)
    N = float(class_sample_count.sum().item())
    p_data = class_sample_count.float() / N  # (K,)

    # target distribution q = (1-lam)*p_data + lam*uniform
    K = float(len(class_sample_count))
    uniform = torch.full_like(p_data, 1.0 / K)
    lam = float(balance_lambda)
    q = (1.0 - lam) * p_data + lam * uniform  # (K,)

    # weights per class = q / p_data
    weights_per_class = q / class_sample_count.float()  # (K,)
    samples_weight = weights_per_class[ancestry_tensor]  # (N,)

    sampler = WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(samples_weight),  # one "epoch" = N draws
        replacement=True,
    )
    return sampler
