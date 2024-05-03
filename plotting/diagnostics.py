import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(model):
    for k, v in model.history.items():
        if k not in ('Expert Losses', 'Model Weights'):
            plt.scatter(np.arange(len(v)), v)
            plt.title(k)
            plt.show()

    expert_loglik = np.array(model.history['Expert Losses'])

    for i, expert_id in enumerate(model.expert_cols):
        plt.scatter(np.arange(expert_loglik.shape[0]), expert_loglik[:, i], label=expert_id)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Expert Losses")
    plt.show()

    model_weights = np.array(model.history['Model Weights'])

    for i, expert_id in enumerate(model.expert_cols):
        plt.scatter(np.arange(model_weights.shape[0]), model_weights[:, i], label=expert_id)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Model Weights")
    plt.show()


def plot_scatter(prs_dataset,
                 model_subset=None,
                 other_predictors=None,
                 sample_mask=None,
                 c=None,
                 vmin=None,
                 vmax=None):

    if model_subset is None:
        model_subset = prs_dataset.expert_ids

    if sample_mask is None:
        sample_mask = np.arange(prs_dataset.N)

    model_idx = [list(prs_dataset.expert_ids).index(x)
                 for x in model_subset if x in prs_dataset.expert_ids]

    pheno = prs_dataset.get_phenotype()[sample_mask]
    m_preds = prs_dataset.get_expert_predictions()[sample_mask, :]

    for m, m_idx in zip(model_subset, model_idx):
        plt.scatter(pheno, m_preds[:, m_idx], c=c, vmin=vmin, vmax=vmax,
                    label=f'{m} (r={np.corrcoef(pheno, m_preds[:, m_idx])[0, 1]:.2}, '
                          f'MSE={np.mean((pheno-m_preds[:, m_idx])**2):.2})',
                    alpha=0.5)

    if other_predictors is not None:
        for m in other_predictors.columns:
            opred = other_predictors[m].values[sample_mask]
            plt.scatter(pheno, opred, c=c, vmin=vmin, vmax=vmax,
                        label=f'{m} (r={np.corrcoef(pheno, opred)[0, 1]:.2}, '
                              f'MSE={np.mean((pheno-opred)**2):.2})',
                        alpha=0.5)

    plt.xlabel("Phenotype")
    plt.ylabel("Predictions")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))