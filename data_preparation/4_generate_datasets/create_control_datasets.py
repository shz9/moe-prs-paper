import argparse
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from magenpy.utils.system_utils import makedir

parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(parent_dir)
sys.path.append(osp.join(parent_dir, "plotting"))

from model.PRSDataset import PRSDataset


def sample_noninformative_prs(
    phenotype,
    biobank,
    n_scores=4,
    keep_scores=None,
    sampling_subset=None,
    r2_threshold=0.001,
    random_state=None,
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
    pgs_df["PGS"] = pgs_df["PGS"].str.replace(
        "_hmPOS_GRCh37|_hmPOS_GRCh38", "", regex=True
    )

    if sampling_subset is not None:
        pgs_df = pgs_df.loc[pgs_df["PGS"].isin(sampling_subset)]

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

    # ---------------------------------------------------------------
    # Quantify the prediction accuracy of all the models:
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

    # ---------------------------------------------------------------

    # Filter scores to only keep ones that fall below the specified threshold:
    acc_results = acc_results.loc[acc_results["R2"] < r2_threshold]

    if len(acc_results) < n_scores:
        raise ValueError(
            f"Only {len(acc_results)} scores with R2 < {r2_threshold}; "
            f"cannot sample n_scores={n_scores}."
        )

    # Sample the number of scores specified by the user:
    acc_results = acc_results.sample(n_scores, random_state=random_state)

    selected_pgs = list(acc_results["Model"].values)

    if keep_scores is not None:
        if isinstance(keep_scores, str):
            keep_scores = [keep_scores]

        assert all([c in pgs_df.columns for c in keep_scores]), (
            "Some keep_scores are not present in the PGS matrix."
        )

        for pgs in keep_scores:
            if pgs not in selected_pgs:
                selected_pgs.append(pgs)

    return selected_pgs


def _as_bool_mask(x):
    return (
        x.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def create_control_prs_dataset(
    biobank,
    analysis_id,
    phenotype,
    selected_pgs,
    pcs_source="1kghdp",
    ancestry_source="1kghdp",
    ancestry_subset=None,
):
    # ------------------------------------------------------------------
    # Phenotype:
    pheno_df = pd.read_csv(
        f"data/phenotypes/{biobank}/{phenotype}.txt",
        sep="\t",
        names=["FID", "IID", phenotype],
    )
    pheno_df.dropna(subset=[phenotype], inplace=True)

    # ------------------------------------------------------------------
    # PGS scores:
    score_long = pd.read_csv(
        f"data/pgsc_calc_scores/{biobank}/{biobank}/score/{biobank}_pgs.txt.gz",
        sep="\t",
        usecols=["sampleset", "FID", "IID", "PGS", "SUM"],
        dtype={"sampleset": str, "FID": str, "IID": str, "PGS": str, "SUM": float},
    )
    score_long = score_long.loc[score_long["sampleset"] == biobank].drop(
        columns=["sampleset"]
    )
    score_long["PGS"] = score_long["PGS"].str.replace(
        "_hmPOS_GRCh37|_hmPOS_GRCh38", "", regex=True
    )

    score_long = score_long.loc[score_long["PGS"].isin(selected_pgs)].copy()
    score_df = score_long.pivot(
        index=["FID", "IID"], columns="PGS", values="SUM"
    ).reset_index()

    missing_pgs = sorted(set(selected_pgs) - set(score_df.columns))
    if missing_pgs:
        raise ValueError(
            f"The following selected scores are missing in score table for {biobank}: {missing_pgs}"
        )

    # Ensure stable column order:
    score_df = score_df[
        ["FID", "IID"] + [c for c in selected_pgs if c in score_df.columns]
    ]

    # Keep merge key dtypes consistent across all sources:
    score_df[["FID", "IID"]] = score_df[["FID", "IID"]].astype(int)
    pheno_df[["FID", "IID"]] = pheno_df[["FID", "IID"]].astype(int)

    # Merge with phenotype:
    score_df = score_df.merge(pheno_df, on=["FID", "IID"])

    # ------------------------------------------------------------------
    # Cluster / ancestry:
    cluster_assignment = pd.read_csv(
        f"data/covariates/{biobank}/cluster_assignment.txt", sep="\t"
    )
    cluster_interp = pd.read_csv(
        f"tables/metadata/{biobank}/cluster_interpretation.csv", index_col=0, header=0
    )
    ancestry_df = pd.read_csv(
        f"data/covariates/{biobank}/{ancestry_source}_ancestry_assignments.txt",
        sep="\t",
        header=0,
    )
    if ancestry_source == "gnomad":
        ancestry_df.rename(columns={"ancestry": "Ancestry"}, inplace=True)

    cluster_assignment[["FID", "IID"]] = cluster_assignment[["FID", "IID"]].astype(int)
    ancestry_df[["FID", "IID"]] = ancestry_df[["FID", "IID"]].astype(int)

    cluster_merged = cluster_assignment.merge(cluster_interp, on="Cluster")
    cluster_merged = cluster_merged.merge(ancestry_df, on=["FID", "IID"], how="right")

    ancestry_cols = [c for c in ancestry_df.columns if c not in ("FID", "IID")]
    cluster_merged = cluster_merged[["FID", "IID", "Description"] + ancestry_cols]
    cluster_merged.rename(columns={"Description": "UMAP_Cluster"}, inplace=True)

    score_df = score_df.merge(cluster_merged, on=["FID", "IID"], how="left")
    score_df["Ancestry"] = score_df["Ancestry"].fillna("OTH")
    score_df["UMAP_Cluster"] = score_df["UMAP_Cluster"].fillna("N/A")
    score_df.fillna(0.0, inplace=True)

    if ancestry_subset is not None:
        if isinstance(ancestry_subset, str):
            ancestry_subset = [ancestry_subset]
        score_df = score_df.loc[score_df["Ancestry"].isin(ancestry_subset)]

    # ------------------------------------------------------------------
    # Covariates:
    covariates_cols = ["Sex"] + ["PC" + str(i + 1) for i in range(10)] + ["Age"]
    covar_df = pd.read_csv(
        f"data/covariates/{biobank}/covars_{pcs_source}_pcs.txt",
        names=["FID", "IID"] + covariates_cols,
        sep="\t",
    )
    covar_df[["FID", "IID"]] = covar_df[["FID", "IID"]].astype(int)
    score_df = score_df.merge(covar_df, on=["FID", "IID"])

    # ------------------------------------------------------------------
    # Cleanup:
    n_before = len(score_df)
    score_df = score_df.dropna().reset_index(drop=True)
    if len(score_df) < n_before:
        print(f"Dropped {n_before - len(score_df)} samples with missing values.")

    if len(score_df) == 0:
        raise ValueError(
            "No samples left after merging phenotype/scores/covariates/ancestry."
        )

    prs_cols = [c for c in selected_pgs if c in score_df.columns]

    return PRSDataset(
        analysis_id=analysis_id,
        dataframe=score_df,
        phenotype_col=phenotype,
        meta_cols=["FID", "IID", "UMAP_Cluster"] + ancestry_cols,
        covariates_cols=covariates_cols,
        prs_cols=prs_cols,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create control analysis table entries using non-informative PRS sampling."
    )
    parser.add_argument(
        "--analysis-file",
        dest="analysis_file",
        type=str,
        required=True,
        help="Path to input analysis table (e.g., multitrait_prs_table.csv).",
    )
    parser.add_argument(
        "--analyses",
        dest="analyses",
        type=str,
        required=True,
        help="Comma-separated list of analysis IDs to process (e.g., CAD_MT,HTN_MT,T2D_MT).",
    )
    parser.add_argument(
        "--biobank",
        dest="biobank",
        type=str,
        required=True,
        choices={"ukbb", "cartagene"},
        help="Biobank used to evaluate non-informative sampling.",
    )
    parser.add_argument(
        "--output-file",
        dest="output_file",
        type=str,
        required=True,
        help="Path to write the control analysis table CSV.",
    )
    parser.add_argument(
        "--sampling-reference-file",
        dest="sampling_reference_file",
        type=str,
        default="tables/multitrait_prs_table.csv",
        help="Table whose unique PGS column defines allowed sampling subset.",
    )
    parser.add_argument(
        "--n-scores",
        dest="n_scores",
        type=int,
        default=4,
        help="Number of non-informative PRS to sample per analysis (before keep_scores union).",
    )
    parser.add_argument(
        "--r2-threshold",
        dest="r2_threshold",
        type=float,
        default=0.001,
        help="Maximum incremental R2 threshold for a PRS to be considered non-informative.",
    )
    parser.add_argument(
        "--control-suffix",
        dest="control_suffix",
        type=str,
        default="_CTRL",
        help="Suffix appended to AnalysisID for control entries.",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=7209,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--disease-flag-col",
        dest="disease_flag_col",
        type=str,
        default="Is_Disease_Matched",
        help="Boolean column in analysis file that marks the disease-matched score.",
    )
    parser.add_argument(
        "--create-harmonized-datasets",
        dest="create_harmonized_datasets",
        action="store_true",
        default=False,
        help="If set, also creates/saves PRSDataset full/train/test for each control analysis.",
    )
    parser.add_argument(
        "--pcs-source",
        dest="pcs_source",
        type=str,
        default="1kghdp",
        choices={"gnomad", "cartagene", "ukbb", "1kghdp"},
        help="PC source for covariates when creating PRSDataset objects.",
    )
    parser.add_argument(
        "--ancestry-source",
        dest="ancestry_source",
        type=str,
        default="1kghdp",
        choices={"gnomad", "1kghdp"},
        help="Ancestry assignment source when creating PRSDataset objects.",
    )
    parser.add_argument(
        "--prop-test",
        dest="prop_test",
        type=float,
        default=0.3,
        help="Proportion of samples to use for test split when creating PRSDataset objects.",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    # ---------------------------------------------------------------
    # Obtain the mapping for the model names:
    from plot_utils import MODEL_NAME_MAP

    model_name_df = pd.DataFrame(
        [(k1, k2, v) for k1, d in MODEL_NAME_MAP.items() for k2, v in d.items()],
        columns=["AnalysisID", "PGS", "Name"],
    ).drop_duplicates(["PGS", "Name"])

    analysis_df = pd.read_csv(args.analysis_file)
    required_cols = {"AnalysisID", "Phenotype_short", "PGS", args.disease_flag_col}
    missing_cols = required_cols - set(analysis_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in analysis file: {missing_cols}")

    sampling_ref_df = pd.read_csv(args.sampling_reference_file)
    if "PGS" not in sampling_ref_df.columns:
        raise ValueError(
            f"Missing PGS column in sampling reference file: {args.sampling_reference_file}"
        )
    sampling_subset = sorted(sampling_ref_df["PGS"].dropna().astype(str).unique())

    requested_analysis_ids = [
        a.strip() for a in args.analyses.split(",") if len(a.strip()) > 0
    ]
    if len(requested_analysis_ids) == 0:
        raise ValueError("No analysis IDs provided in --analyses.")

    # Prefer names from the input analysis file, then fill with MODEL_NAME_MAP names.
    pgs_name_map = (
        analysis_df[["PGS", "PGS_Name"]]
        .dropna()
        .drop_duplicates("PGS")
        .set_index("PGS")["PGS_Name"]
        .to_dict()
        if "PGS_Name" in analysis_df.columns
        else {}
    )
    for _, row in model_name_df.drop_duplicates("PGS").iterrows():
        pgs_name_map.setdefault(row["PGS"], row["Name"])

    control_rows = []

    for i, analysis_id in enumerate(requested_analysis_ids):
        adf = analysis_df.loc[analysis_df["AnalysisID"] == analysis_id].copy()
        if adf.empty:
            raise ValueError(f"AnalysisID '{analysis_id}' not found in {args.analysis_file}")

        phenotype = str(adf["Phenotype_short"].iloc[0])

        disease_mask = _as_bool_mask(adf[args.disease_flag_col])
        n_disease = int(disease_mask.sum())
        if n_disease != 1:
            raise ValueError(
                f"Expected exactly 1 disease-matched score for {analysis_id} "
                f"in column '{args.disease_flag_col}', found {n_disease}."
            )
        disease_row = adf.loc[disease_mask].iloc[0]

        disease_pgs = str(disease_row["PGS"])

        selected_pgs = sample_noninformative_prs(
            phenotype=phenotype,
            biobank=args.biobank,
            n_scores=args.n_scores,
            keep_scores=[disease_pgs],
            sampling_subset=sampling_subset,
            r2_threshold=args.r2_threshold,
            random_state=args.seed + i,
        )

        template_row = adf.iloc[0].to_dict()
        control_analysis_id = (
            analysis_id if args.control_suffix == "" else f"{analysis_id}{args.control_suffix}"
        )

        if args.create_harmonized_datasets:
            print(
                f"> Creating control PRSDataset for {control_analysis_id} ({args.biobank}) "
                f"with {len(selected_pgs)} scores."
            )
            prs_dataset = create_control_prs_dataset(
                biobank=args.biobank,
                analysis_id=control_analysis_id,
                phenotype=phenotype,
                selected_pgs=selected_pgs,
                pcs_source=args.pcs_source,
                ancestry_source=args.ancestry_source,
            )

            out_dir = f"data/harmonized_data/{control_analysis_id}/{args.biobank}/"
            makedir(out_dir)

            prs_dataset.save(osp.join(out_dir, "full_data.pkl"))

            np.random.seed(args.seed + i)
            train_data, test_data = prs_dataset.train_test_split(test_size=args.prop_test)
            train_data.save(osp.join(out_dir, "train_data.pkl"))
            test_data.save(osp.join(out_dir, "test_data.pkl"))

        for pgs in selected_pgs:
            row = template_row.copy()
            row["AnalysisID"] = control_analysis_id
            row["PGS"] = pgs
            row[args.disease_flag_col] = bool(pgs == disease_pgs)
            if "PGSCatalog_ID" in row:
                row["PGSCatalog_ID"] = pgs
            if "PGS_Name" in row:
                row["PGS_Name"] = pgs_name_map.get(pgs, pgs)
            if "Notes" in row:
                if pgs == disease_pgs:
                    row["Notes"] = "Disease-specific score retained in control analysis"
                else:
                    row["Notes"] = (
                        f"Sampled non-informative control score "
                        f"(biobank={args.biobank}, incremental R2<{args.r2_threshold})"
                    )

            control_rows.append(row)

    if len(control_rows) == 0:
        raise ValueError("No control rows were generated.")

    control_df = pd.DataFrame(control_rows)

    # Keep the same column order as the input analysis table.
    ordered_cols = [c for c in analysis_df.columns if c in control_df.columns]
    control_df = control_df[ordered_cols]

    output_dir = osp.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    control_df.to_csv(args.output_file, index=False)
    print(
        f"Saved {len(control_df)} control rows for {len(requested_analysis_ids)} analyses "
        f"to {args.output_file}"
    )
