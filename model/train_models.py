import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
from baseline_models import MultiPRS, AncestryWeightedPRS
from moe import MoEPRS
from moe_pytorch import Lit_MoEPRS, train_model
from PRSDataset import PRSDataset
from plotting.plot_utils import MODEL_NAME_GNOMAD_ANCESTRY_MAP
from magenpy.utils.system_utils import makedir
import argparse


def train_baseline_linear_models(dataset,
                                 penalty_type=None,
                                 penalty=0.,
                                 class_weights=None,
                                 add_intercept=True):

    dataset.set_backend("numpy")

    print(f"> Training baseline models for {dataset.phenotype_col} with {dataset.N} samples...")

    base_models = dict()

    base_models['MultiPRS'] = MultiPRS(prs_dataset=dataset,
                                       expert_cols=dataset.prs_cols,
                                       covariates_cols=dataset.covariates_cols,
                                       add_intercept=add_intercept,
                                       class_weights=class_weights,
                                       penalty_type=penalty_type,
                                       penalty=penalty)

    base_models['MultiPRS'].fit()

    try:
        amw_b = AncestryWeightedPRS(prs_dataset=dataset,
                                    expert_cols=dataset.prs_cols,
                                    covariates_cols=dataset.covariates_cols,
                                    weighing_scheme='before',
                                    expert_ancestry_map=MODEL_NAME_GNOMAD_ANCESTRY_MAP,
                                    add_intercept=add_intercept,
                                    class_weights=class_weights,
                                    penalty_type=penalty_type,
                                    penalty=penalty)
        amw_b.fit()

        base_models['AncestryWeightedPRS_B'] = amw_b
    except Exception as e:
        pass

    try:
        amw_a = AncestryWeightedPRS(prs_dataset=dataset,
                                    expert_cols=dataset.prs_cols,
                                    covariates_cols=dataset.covariates_cols,
                                    weighing_scheme='after',
                                    expert_ancestry_map=MODEL_NAME_GNOMAD_ANCESTRY_MAP,
                                    add_intercept=add_intercept,
                                    class_weights=class_weights,
                                    penalty_type=penalty_type,
                                    penalty=penalty)
        amw_a.fit()

        base_models['AncestryWeightedPRS_A'] = amw_a
    except Exception as e:
        pass

    base_models['Covariates'] = MultiPRS(prs_dataset=dataset,
                                         covariates_cols=dataset.covariates_cols,
                                         add_intercept=add_intercept,
                                         class_weights=class_weights,
                                         penalty_type=penalty_type,
                                         penalty=penalty)

    base_models['Covariates'].fit()

    for i, pgs_id in enumerate(dataset.prs_cols):

        base_models[f'{pgs_id}-covariates'] = MultiPRS(prs_dataset=dataset,
                                                       expert_cols=pgs_id,
                                                       covariates_cols=dataset.covariates_cols,
                                                       add_intercept=add_intercept,
                                                       class_weights=class_weights,
                                                       penalty_type=penalty_type,
                                                       penalty=penalty)

        base_models[f'{pgs_id}-covariates'].fit()

    return base_models


def train_moe_model_numpy(dataset,
                          fix_residuals=False,
                          class_weights=None,
                          gate_penalty=0.,
                          expert_penalty=0.,
                          gate_add_intercept=True,
                          expert_add_intercept=True,
                          optimizer='L-BFGS-B'):

    print(f"> Training MoE model for {dataset.phenotype_col} with {dataset.N} samples...")

    dataset.set_backend("numpy")

    """
    m1 = MoEPRS(prs_dataset=dataset,
                expert_cols=dataset.prs_cols,
                gate_input_cols=dataset.covariates_cols,
                expert_covariates_cols=dataset.covariates_cols,
                optimizer=optimizer,
                fix_residuals=fix_residuals,
                class_weights=class_weights,
                gate_add_intercept=gate_add_intercept,
                expert_add_intercept=expert_add_intercept,
                gate_penalty=gate_penalty,
                expert_penalty=expert_penalty,
                n_jobs=min(4, dataset.n_prs_models))
    m1.fit()

    # Model without covariates in experts:
    m2 = MoEPRS(prs_dataset=dataset,
                expert_cols=dataset.prs_cols,
                gate_input_cols=dataset.covariates_cols,
                optimizer=optimizer,
                fix_residuals=fix_residuals,
                class_weights=class_weights,
                gate_add_intercept=gate_add_intercept,
                expert_add_intercept=expert_add_intercept,
                gate_penalty=gate_penalty,
                expert_penalty=expert_penalty,
                n_jobs=min(4, dataset.n_prs_models))
    m2.fit()
    """

    m3 = MoEPRS(prs_dataset=dataset,
                expert_cols=[c for c in dataset.prs_cols if c != 'Random'],
                gate_input_cols=dataset.covariates_cols,
                expert_covariates_cols=dataset.covariates_cols,
                optimizer=optimizer,
                fix_residuals=fix_residuals,
                class_weights=class_weights,
                gate_add_intercept=gate_add_intercept,
                expert_add_intercept=expert_add_intercept,
                gate_penalty=gate_penalty,
                expert_penalty=expert_penalty,
                n_jobs=min(4, dataset.n_prs_models))
    m3.fit()

    m4 = MoEPRS(prs_dataset=dataset,
                expert_cols=[c for c in dataset.prs_cols if c != 'Random'],
                gate_input_cols=dataset.covariates_cols,
                optimizer=optimizer,
                fix_residuals=fix_residuals,
                class_weights=class_weights,
                gate_add_intercept=gate_add_intercept,
                expert_add_intercept=expert_add_intercept,
                gate_penalty=gate_penalty,
                expert_penalty=expert_penalty,
                n_jobs=min(4, dataset.n_prs_models))
    m4.fit()

    m5 = MoEPRS(prs_dataset=dataset,
                expert_cols=[c for c in dataset.prs_cols if c != 'Random'],
                gate_input_cols=dataset.covariates_cols,
                optimizer=optimizer,
                fix_residuals=fix_residuals,
                class_weights=class_weights,
                gate_add_intercept=gate_add_intercept,
                expert_add_intercept=False,
                gate_penalty=gate_penalty,
                expert_penalty=expert_penalty,
                n_jobs=min(4, dataset.n_prs_models))
    m5.fit()

    return {
        #'MoE': m1,
        #'MoE-no-cov': m2,
        'MoE-no-rand': m3,
        'MoE-no-cov-rand': m4,
        'MoE-no-cov-rand-int': m5
    }


def train_moe_models_torch(dataset,
                           gate_model_layers=None,
                           add_covariates_to_experts=False,
                           loss='likelihood_mixture',
                           optimizer='Adam',
                           penalty=0.,
                           learning_rate=1e-2,
                           max_epochs=100,
                           batch_size=None,
                           weigh_samples=False):

    dataset.set_backend("torch")

    dataset.group_getitem_cols = {
        'phenotype': [dataset.phenotype_col],
        'gate_input': dataset.covariates_cols,
        'experts': dataset.prs_cols
    }

    if add_covariates_to_experts:
        dataset.group_getitem_cols['expert_covariates'] = dataset.covariates_cols

    # Initialize the torch MoE model:
    m = Lit_MoEPRS(dataset.group_getitem_cols,
                   gate_model_layers=gate_model_layers,
                   loss=loss,
                   family=dataset.phenotype_likelihood,
                   optimizer=optimizer,
                   learning_rate=learning_rate,
                   weight_decay=penalty)

    # Train with PyTorch Lightning:
    _, m = train_model(m,
                       dataset,
                       max_epochs=max_epochs,
                       batch_size=batch_size,
                       weigh_samples=weigh_samples)

    return m


def train_all_models(dataset,
                     baseline_kwargs,
                     moe_kwargs,
                     skip_baseline=False,
                     skip_moe=False,
                     moe_pytorch_kwargs=None):

    trained_models = {}

    if not skip_baseline:
        trained_models.update(train_baseline_linear_models(dataset, **baseline_kwargs))

    if not skip_moe:
        trained_models.update(train_moe_model_numpy(dataset, **moe_kwargs))

    #if moe_pytorch_kwargs is not None:
    #    moe_pytorch_model = train_moe_models_torch(dataset, **moe_pytorch_kwargs)
    #    trained_models.update({'MoE-PyTorch': moe_pytorch_model})

    return trained_models


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train baseline and MoE models.')
    parser.add_argument('--dataset-path', dest='dataset_path', type=str, required=True,
                        help='The path to the dataset file.')
    parser.add_argument('--baseline-kwargs', dest='baseline_kwargs', type=str, default='',
                        help='A comma-separated list of key-value pairs with the arguments for the baseline models.')
    parser.add_argument('--moe-kwargs', dest='moe_kwargs', type=str, default='',
                        help='A comma-separated list of key-value pairs with the arguments for the MoE model.')
    parser.add_argument('--residualize-phenotype', dest='residualize_phenotype', action='store_true',
                        default=False,
                        help='Whether to residualize the phenotype before training the models.')
    parser.add_argument('--residualize-prs', dest='residualize_prs', action='store_true',
                        default=False,
                        help='Whether to residualize the PRS before training the models.')
    parser.add_argument('--skip-baseline', dest='skip_baseline', action='store_true',
                        default=False,
                        help='Whether to skip training the baseline models.')
    parser.add_argument('--skip-moe', dest='skip_moe', action='store_true',
                        default=False,
                        help='Whether to skip training the MoE models.')
    parser.add_argument('--skip-moe-pytorch', dest='skip_moe_pytorch', action='store_true',
                        default=False,
                        help='Whether to skip training the MoE models with PyTorch.')
    args = parser.parse_args()

    prs_dataset = PRSDataset.from_pickle(args.dataset_path)

    if args.residualize_phenotype:
        try:
            prs_dataset.adjust_phenotype_for_covariates()
        except AssertionError:
            print("Could not residualize the phenotype.")

    if args.residualize_prs:
        prs_dataset.adjust_prs_for_covariates()

    baseline_kwargs = {}
    if len(args.baseline_kwargs) > 0:
        baseline_kwargs = {k: v for k, v in [kw.split('=') for kw in args.baseline_kwargs.split(',')] if v}
    moe_kwargs = {}
    if len(args.moe_kwargs) > 0:
        moe_kwargs = {k: v for k, v in [kw.split('=') for kw in args.moe_kwargs.split(',')] if v}

    trained_models = train_all_models(prs_dataset, baseline_kwargs, moe_kwargs,
                                      skip_baseline=args.skip_baseline,
                                      skip_moe=args.skip_moe)

    output_dir = osp.dirname(args.dataset_path).replace('harmonized_data', 'trained_models')
    dataset_name = osp.basename(args.dataset_path).replace('.pkl', '')

    if args.residualize_phenotype:
        dataset_name += '_rph'
    if args.residualize_prs:
        dataset_name += '_rprs'

    output_dir = osp.join(output_dir, dataset_name)

    makedir(output_dir)

    print("> Saving trained models to:\n\t", output_dir)

    for model_name, model in trained_models.items():
        model.save(osp.join(output_dir, f'{model_name}.pkl'))
