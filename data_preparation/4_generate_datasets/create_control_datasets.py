import numpy as np
import pandas as pd


def construct_control_datasets(
    phenotype, biobank, n_scores=5, keep_prs=None, r2_threshold=0.005
):
    # ---------------------------------------------------------------------------------
    # Read the PGS scores:
    pgs_df = pd.read_csv(
        f"data/pgsc_calc_scores/{biobank}/{biobank}/score/{biobank}_pgs.txt.gz",
        sep="\t",
        usecols=["sampleset", "FID", "IID", "PGS", "SUM"],
        dtype={"sampleset": str, "FID": str, "IID": str, "PGS": str, "SUM": float},
    )
    pgs_df = pgs_df.loc[pgs_df.sampleset == biobank].drop(columns=["sampleset"])

    # Rename the PGSs to remove information related to the genome build:
    pgs_df["PGS"] = (
        pgs_df["PGS"].str.replace("_hmPOS_GRCh37", "").replace("_hmPOS_GRCh38", "")
    )

    if keep_prs is not None:
        pgs_df = pgs_df.loc[pgs_df["PGS"].isin(keep_prs)]

        if len(np.unique(pgs_df["PGS"])) < n_scores:
            raise ValueError(f"There are less than {n_scores} in the dataset.")

    # Pivot the PGS dataframe:
    pgs_df = pgs_df.pivot(
        index=["FID", "IID"], columns="PGS", values="SUM"
    ).reset_index()

    # ---------------------------------------------------------------------------------
    # Read the phenotype table:

    df_pheno = pd.read_csv(
        f"data/phenotypes/{biobank}/{phenotype}.txt",
        sep="\t",
        names=["FID", "IID", "phenotype"],
    )

    # ---------------------------------------------------------------------------------
    # Read the table of covariates (to compute incremental R^2):

    covariates_cols = ["Sex"] + ["PC" + str(i + 1) for i in range(10)] + ["Age"]

    covar_df = pd.read_csv(
        f"data/covariates/{biobank}/covars_1kghdp_pcs.txt",
        names=["FID", "IID"] + covariates_cols,
        sep="\t",
    )

    # ---------------------------------------------------------------------------------
    # Merge the three data sources together:

    covar_df[["FID", "IID"]] = covar_df[["FID", "IID"]].astype(int)
    pgs_df[["FID", "IID"]] = pgs_df[["FID", "IID"]].astype(int)
    df_pheno[["FID", "IID"]] = df_pheno[["FID", "IID"]].astype(int)

    m_df = pgs_df.merge(df_pheno).merge(covar_df)

    from viprs.eval.continuous_metrics import incremental_r2

    acc_results = []

    for c in pgs_df.columns[2:]:
        acc_results.append(
            {
                "Model": c,
                "R2": incremental_r2(
                    m_df["phenotype"].values, m_df[c].values, m_df[covariates_cols]
                ),
            }
        )

    acc_results = pd.DataFrame(acc_results)

    from plot_utils import MODEL_NAME_MAP

    model_name_df = pd.DataFrame(
        [(k1, k2, v) for k1, d in MODEL_NAME_MAP.items() for k2, v in d.items()],
        columns=["AnalysisID", "PGS", "Name"],
    ).drop_duplicates(["PGS", "Name"])

    return acc_results.merge(
        model_name_df, left_on="Model", right_on="PGS"
    ).sort_values("R2")

    # acc_results = acc_results.loc[acc_results["R2"] < r2_threshold]
