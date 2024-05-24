import pandas as pd

MODEL_NAME_GNOMAD_ANCESTRY_MAP = {
    'PGS003843': 'nfe',
    'PGS003845': 'afr',
    'PGS002313': 'nfe',
    'PGS002360': 'eas',

    'PGS003864': 'amr',
    'PGS003862': 'nfe',
    'PGS003863': 'afr',
    'PGS003865': 'eas',

    'PGS002800': 'sas',
    'PGS002801': 'afr',
    'PGS002803': 'eas',
    'PGS002804': 'nfe',
    'PGS002805': 'amr',

    'PGS000894': 'amr',
    'PGS000896': 'sas',
    'PGS000890': 'eas',
    'PGS000886': 'afr',
    'PGS000892': 'nfe',

    'PGS003770': 'afr',
    'PGS003775': 'eas',
    'PGS003780': 'sas',
    'PGS003768': 'nfe',

    'PGS000806': 'afr',
    'PGS000805': 'nfe',
    'PGS000808': 'amr',

    'PGS002311': 'nfe',
    'PGS002358': 'eas',
}

df = pd.read_csv("tables/phenotype_prs_table.csv")
MODEL_NAME_MAP = dict(zip(df['PGSCatalog_ID'], df['Training_cohort']))


GROUP_MAP = {
    'afr': 'AFR',
    'eas': 'EAS',
    'mid': 'MID',
    'nfe': 'EUR',
    'sas': 'SAS',
    'asj': 'ASJ',
    'ami': 'AMI',
    'amr': 'AMR',
    'fin': 'FIN',
    'oth': 'OTH',
    '0': 'Female',
    '1': 'Male'
}

SORTED_ANCESTRY_LABEL = ['All', 'EUR', 'MID', 'FIN', 'ASJ', 'AMR', 'OTH', 'SAS', 'EAS', 'AFR']

UKBB_SORTED_UMAP_CLUSTERS = [
    'All',
    '17 ENG-BRI',
    '20 ENG-BRI',
    '21 ENG-BRI-OTH',
    '24 ENG-BRI-OTH',
    '16 ESPPOR',
    '1 ITA',
    '11 ENG-MIX',
    '3 ENG-BRI',
    '5 LEV',
    '9 SAS-MIX',
    '6 NAF',
    '2 FIN',
    '4 ENG-BRI-AOW',
    '25 ENG-AFR-CAR-MIX',
    '7 SAS',
    '23 HAFR',
    '14 ENG-EAS-MIX',
    '22 ENG-CAR-WAB',
    '8 SAS-IND',
    '10 SOM',
    '12 AMR',
    '15 NEP',
    '13 SEA-CHN-OTH',
    '0 JPN',
    '19 WAFR-CAR',
    '18 AFR'
]

CARTAGENE_SORTED_UMAP_CLUSTERS = [
    'All',
    '14-FRC',
    '13-FRC',
    '10-CAN-FRC',
    '12-CAN-FRC',
    '11-MED',
    '7-EER',
    '9-EUR-JEW',
    '5-MIE',
    '4-NAF',
    '8-EER-JEW',
    '2-SAS',
    '6-AFR-EUR',
    '3-CSA',
    '1-EAS',
    '0-HAI-CAR'
]


def assign_models_consistent_colors(models, palette='Set2'):
    """
    Assign consistent colors to the models for plotting.
    :param models: A list of model names
    :param palette: The color palette to use
    :return: A dictionary of model names and colors
    """
    import seaborn as sns

    baseline_ancestry_model_names = ['ALL', 'AFR', 'AMR', 'EAS', 'EUR', 'SAS', 'Random']
    ancestry_colors = sns.color_palette(palette, len(baseline_ancestry_model_names))

    colors = dict(zip(baseline_ancestry_model_names, ancestry_colors))

    remaining_models = sorted(list(set(MODEL_NAME_MAP.values()) - set(baseline_ancestry_model_names)))
    remaining_colors = sns.color_palette(palette, len(remaining_models))

    colors.update(dict(zip(remaining_models, remaining_colors)))

    return {m: colors[m] for m in models}


def sort_groups(groups):

    if len(set(groups).intersection(SORTED_ANCESTRY_LABEL)) > 2:
        return sorted(groups, key=lambda x: SORTED_ANCESTRY_LABEL.index(x))
    elif len(set(groups).intersection(UKBB_SORTED_UMAP_CLUSTERS)) > 2:
        return sorted(groups, key=lambda x: UKBB_SORTED_UMAP_CLUSTERS.index(x))
    elif len(set(groups).intersection(CARTAGENE_SORTED_UMAP_CLUSTERS)) > 2:
        return sorted(groups, key=lambda x: CARTAGENE_SORTED_UMAP_CLUSTERS.index(x))
    else:
        return sorted(groups)


def read_transform_eval_metrics(file_path):
    """
    Read the evaluation metrics from a CSV file and transform the names
    of the models + the phenotype for the purposes of plotting.
    """

    eval_df = pd.read_csv(file_path)
    phenotype_id = file_path.split('/')[-3]

    eval_df['Phenotype'] = phenotype_id
    eval_df['Test biobank'] = file_path.split('/')[-2].upper()

    def map_model_name(x):
        try:
            biobank, rest = x.split('/')
            _, rest = rest.split(':')

            if rest in (pd.Series(MODEL_NAME_MAP.keys()) + '-covariates').values:
                m = rest.replace('-covariates', '')
            else:
                m = rest

            try:
                return MODEL_NAME_MAP[m] + f' ({biobank})'
            except KeyError as e:
                return m + f' ({biobank})'
        except ValueError as e:
            return x

    def map_dataset_name(x):
        try:
            split_x = x.split(':')
            if len(split_x) > 1:
                return split_x[0]
            else:
                return None
        except Exception as e:
            print(e)
            return x

    def assign_training_biobank(x):
        try:
            split_x = x.split('/')
            if len(split_x) > 1:
                return split_x[0].upper()
            else:
                return None
        except Exception as e:
            print(e)
            return x

    def assign_model_category(x):
        if 'MoE' in x:
            return 'MoE'
        elif 'MultiPRS' in x:
            return 'MultiPRS'
        elif 'AncestryWeightedPRS' in x:
            return 'AncestryWeightedPRS'
        elif 'Covariates' in x:
            return 'Covariates'
        elif 'Random' in x:
            return 'Random'
        else:
            if 'covariates' in x:
                return 'SinglePRS+Covariates'
            else:
                return 'SinglePRS'

    eval_df['Training biobank'] = eval_df['PGS'].apply(assign_training_biobank)
    eval_df['Training dataset'] = eval_df['PGS'].apply(map_dataset_name)
    eval_df['Model Name'] = eval_df['PGS'].apply(map_model_name)
    eval_df['Model Category'] = eval_df['PGS'].apply(assign_model_category)

    def map_group_name(x):
        try:
            return GROUP_MAP[x]
        except KeyError:
            return x

    eval_df['EvalGroup'] = eval_df['EvalGroup'].apply(map_group_name)
    eval_df.rename(columns={'EvalGroup': 'Evaluation Group'}, inplace=True)

    return eval_df
