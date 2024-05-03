import pandas as pd

covar_df = pd.read_csv("/Users/szabad/moe_project/data/covariates/cartagene.txt", sep="\t",
                      names=['FID', 'IID', 'Sex'] + ['PC' + str(i + 1) for i in range(10)] + ['Age'])

gnomad_df = pd.read_csv("/Users/szabad/moe_project/data/covariates/cartagene/projected_sample_pcs.txt", sep="\t")
gnomad_df['FID'] = 0

covar_df = covar_df.merge(gnomad_df, on=['FID', 'IID'])

covar_df[['FID', 'IID', 'Sex'] + [f'PC_{i}' for i in range(1, 11)] + ['Age']].to_csv(
    "/Users/szabad/moe_project/data/covariates/cartagene/covars_gnomad_pcs.txt",
    sep="\t", header=False, index=False)


covar_df[['FID', 'IID', 'Sex'] + [f'PC{i}' for i in range(1, 11)] + ['Age']].to_csv(
    "/Users/szabad/moe_project/data/covariates/cartagene/covars_cartagene_pcs.txt",
    sep="\t", header=False, index=False)
