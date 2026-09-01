import argparse
import copy
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
    return_details=False,
    analysis_id=None,
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
    from viprs.eval.binary_metrics import liability_r2
    from viprs.eval.continuous_metrics import incremental_r2

    phenotype_vals = pd.to_numeric(m_df["phenotype"], errors="coerce").values
    valid_pheno = ~np.isnan(phenotype_vals)
    unique_pheno = np.unique(phenotype_vals[valid_pheno])

    is_binary = len(unique_pheno) == 2
    if is_binary:
        # Robust 0/1 recoding for binary phenotypes (e.g. supports 1/2 coding).
        unique_sorted = np.sort(unique_pheno)
        phenotype_eval = np.where(phenotype_vals == unique_sorted[-1], 1.0, 0.0)
        r2_metric_name = "Liability_R2"
    else:
        phenotype_eval = phenotype_vals
        r2_metric_name = "Incremental_R2"

    acc_results = []

    for c in pgs_df.columns[2:]:
        try:
            prs_vals = pd.to_numeric(m_df[c], errors="coerce").values
            keep = valid_pheno & (~np.isnan(prs_vals))

            if is_binary:
                r2_val = liability_r2(
                    phenotype_eval[keep], prs_vals[keep], m_df.loc[keep, covariates_cols]
                )
            else:
                r2_val = incremental_r2(
                    phenotype_eval[keep], prs_vals[keep], m_df.loc[keep, covariates_cols]
                )
        except Exception:
            r2_val = np.nan

        acc_results.append({"Model": c, "R2": r2_val})

    acc_results = pd.DataFrame(acc_results)
    acc_results = acc_results.loc[~acc_results["R2"].isna()].copy()

    # ---------------------------------------------------------------

    # Filter scores to only keep ones that fall below the specified threshold:
    eligible_results = acc_results.loc[acc_results["R2"] < r2_threshold].copy()

    if len(eligible_results) < n_scores:
        context = (
            f" for analysis_id={analysis_id}, phenotype={phenotype}, biobank={biobank}"
            if analysis_id is not None
            else f" for phenotype={phenotype}, biobank={biobank}"
        )
        lowest_results = acc_results.sort_values("R2").head(10)
        lowest_msg = ", ".join(
            f"{row.Model}={row.R2:.4g}" for row in lowest_results.itertuples()
        )
        raise ValueError(
            f"Only {len(eligible_results)} scores with R2 < {r2_threshold}{context}; "
            f"cannot sample n_scores={n_scores}. "
            f"Lowest observed R2 values: {lowest_msg}"
        )

    # Sample the number of scores specified by the user:
    sampled_results = eligible_results.sample(n_scores, random_state=random_state)

    selected_pgs = list(sampled_results["Model"].values)

    if keep_scores is not None:
        if isinstance(keep_scores, str):
            keep_scores = [keep_scores]

        assert all([c in pgs_df.columns for c in keep_scores]), (
            "Some keep_scores are not present in the PGS matrix."
        )

        for pgs in keep_scores:
            if pgs not in selected_pgs:
                selected_pgs.append(pgs)

    if not return_details:
        return selected_pgs

    sampled_set = set(sampled_results["Model"].values)
    selected_details = (
        acc_results.set_index("Model")
        .reindex(selected_pgs)
        .reset_index()
        .rename(columns={"index": "Model"})
    )
    selected_details["SelectedByThreshold"] = selected_details["Model"].isin(
        sampled_set
    )
    selected_details["R2_Threshold"] = float(r2_threshold)
    selected_details["R2_Metric"] = r2_metric_name

    return selected_pgs, selected_details


def _as_bool_mask(x):
    return x.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _normalize_sample_id_col(x):
    x_str = x.astype(str).str.strip()
    x_num = pd.to_numeric(x_str, errors="coerce")

    if x_num.notna().all() and np.all(np.isclose(x_num, np.round(x_num))):
        return x_num.astype(np.int64).astype(str)

    return x_str


def _sample_key_df(prs_dataset):
    required_cols = ["FID", "IID"]
    missing_cols = [c for c in required_cols if c not in prs_dataset.data.columns]
    if missing_cols:
        raise ValueError(
            f"PRSDataset for {prs_dataset.analysis_id} is missing sample ID columns: "
            f"{missing_cols}"
        )

    key_df = prs_dataset.data[required_cols].copy()
    for col in required_cols:
        key_df[col] = _normalize_sample_id_col(key_df[col])

    duplicated = key_df.duplicated()
    if duplicated.any():
        duplicate_examples = key_df.loc[duplicated, required_cols].head(5)
        raise ValueError(
            f"PRSDataset for {prs_dataset.analysis_id} contains duplicate FID/IID "
            f"sample identifiers. Examples: {duplicate_examples.to_dict('records')}"
        )

    return key_df


def _sample_key_set(prs_dataset):
    key_df = _sample_key_df(prs_dataset)
    return set(map(tuple, key_df[["FID", "IID"]].to_numpy()))


def _format_sample_key_examples(sample_keys, n=5):
    return [{"FID": fid, "IID": iid} for fid, iid in sorted(sample_keys)[:n]]


def _fold_names(n_folds):
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    return [f"fold_{fold_idx}" for fold_idx in range(1, n_folds + 1)]


def _reference_split_paths(reference_analysis_id, biobank, fold):
    ref_dir = f"data/harmonized_data/{reference_analysis_id}/{biobank}/{fold}"
    return [
        osp.join(ref_dir, "train_data.pkl"),
        osp.join(ref_dir, "test_data.pkl"),
    ]


def _missing_reference_split_paths(reference_analysis_id, biobank, n_folds):
    return [
        p
        for fold in _fold_names(n_folds)
        for p in _reference_split_paths(reference_analysis_id, biobank, fold)
        if not osp.exists(p)
    ]


def split_like_reference_dataset(
    prs_dataset, reference_analysis_id, biobank, fold
):
    ref_train_path, ref_test_path = _reference_split_paths(
        reference_analysis_id, biobank, fold
    )
    missing_paths = [
        p for p in (ref_train_path, ref_test_path) if not osp.exists(p)
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"Cannot match reference cross-validation {fold} because the "
            f"following dataset files are missing: {missing_paths}"
        )

    ref_train_data = PRSDataset.from_pickle(ref_train_path)
    ref_test_data = PRSDataset.from_pickle(ref_test_path)

    train_keys = _sample_key_set(ref_train_data)
    test_keys = _sample_key_set(ref_test_data)

    split_overlap = train_keys & test_keys
    if split_overlap:
        raise ValueError(
            f"Reference {fold} for {reference_analysis_id} ({biobank}) has "
            f"{len(split_overlap)} samples present in both train and test. "
            f"Examples: {_format_sample_key_examples(split_overlap)}"
        )

    reference_keys = train_keys | test_keys
    control_key_df = _sample_key_df(prs_dataset)
    control_key_tuples = list(map(tuple, control_key_df[["FID", "IID"]].to_numpy()))
    control_keys = set(control_key_tuples)

    missing_from_reference = control_keys - reference_keys
    missing_from_control = reference_keys - control_keys

    if missing_from_reference or missing_from_control:
        details = []
        if missing_from_reference:
            details.append(
                f"{len(missing_from_reference)} control samples are absent from "
                f"the reference split; examples: "
                f"{_format_sample_key_examples(missing_from_reference)}"
            )
        if missing_from_control:
            details.append(
                f"{len(missing_from_control)} reference samples are absent from "
                f"the control dataset; examples: "
                f"{_format_sample_key_examples(missing_from_control)}"
            )
        raise ValueError(
            f"Control dataset for {prs_dataset.analysis_id} ({biobank}) does not "
            f"match the sample set of reference analysis {reference_analysis_id}. "
            + " ".join(details)
        )

    train_mask = pd.Series(
        [sample_key in train_keys for sample_key in control_key_tuples],
        index=control_key_df.index,
    )
    test_mask = pd.Series(
        [sample_key in test_keys for sample_key in control_key_tuples],
        index=control_key_df.index,
    )

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Matched {fold} for {prs_dataset.analysis_id} ({biobank}) produced "
            f"{int(train_mask.sum())} train and {int(test_mask.sum())} test samples."
        )

    train_dataset = copy.deepcopy(prs_dataset)
    train_dataset.filter_samples(train_mask)

    test_dataset = copy.deepcopy(prs_dataset)
    test_dataset.filter_samples(test_mask)

    return train_dataset, test_dataset


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
        default=0.005,
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
        help=(
            "If set, also saves a full PRSDataset and fold-specific train/test "
            "datasets for each control analysis."
        ),
    )
    parser.add_argument(
        "--match-reference-split",
        dest="match_reference_split",
        action="store_true",
        default=False,
        help=(
            "When creating harmonized control datasets, reuse every K-fold "
            "sample split from the corresponding non-control analysis dataset."
        ),
    )
    parser.add_argument(
        "--target-biobanks",
        dest="target_biobanks",
        nargs="+",
        type=str,
        default=None,
        choices={"ukbb", "cartagene"},
        help=(
            "Biobank list for control harmonized dataset creation when "
            "--create-harmonized-datasets is enabled. Defaults to --biobank."
        ),
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
        "--n-folds",
        "--k-folds",
        dest="n_folds",
        type=int,
        default=10,
        help="Number of cross-validation folds to create (default: 10).",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.match_reference_split and not args.create_harmonized_datasets:
        raise ValueError(
            "--match-reference-split requires --create-harmonized-datasets."
        )
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")

    if args.create_harmonized_datasets:
        if args.target_biobanks is None:
            target_biobanks = [args.biobank]
        else:
            target_biobanks = sorted(set(args.target_biobanks))
    else:
        target_biobanks = []

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
            raise ValueError(
                f"AnalysisID '{analysis_id}' not found in {args.analysis_file}"
            )

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

        print(
            f"> Sampling non-informative PRSs for {analysis_id} "
            f"({phenotype}, {args.biobank})."
        )
        selected_pgs, selected_details = sample_noninformative_prs(
            phenotype=phenotype,
            biobank=args.biobank,
            n_scores=args.n_scores,
            keep_scores=[disease_pgs],
            sampling_subset=sampling_subset,
            r2_threshold=args.r2_threshold,
            random_state=args.seed + i,
            return_details=True,
            analysis_id=analysis_id,
        )
        sampled_r2_map = dict(
            zip(selected_details["Model"].values, selected_details["R2"].values)
        )
        sampled_metric_map = dict(
            zip(selected_details["Model"].values, selected_details["R2_Metric"].values)
        )
        sampled_flag_map = dict(
            zip(
                selected_details["Model"].values,
                selected_details["SelectedByThreshold"].values,
            )
        )

        template_row = adf.iloc[0].to_dict()
        control_analysis_id = (
            analysis_id
            if args.control_suffix == ""
            else f"{analysis_id}{args.control_suffix}"
        )

        for biobank in target_biobanks:
            if args.match_reference_split:
                missing_reference_paths = _missing_reference_split_paths(
                    analysis_id, biobank, args.n_folds
                )
                if missing_reference_paths:
                    print(
                        f"> Skipping control PRSDataset for {control_analysis_id} "
                        f"({biobank}) because one or more reference folds for "
                        f"{analysis_id} are unavailable. Missing files: "
                        f"{missing_reference_paths}"
                    )
                    continue

            print(
                f"> Creating control PRSDataset for {control_analysis_id} ({biobank}) "
                f"with {len(selected_pgs)} scores."
            )
            try:
                prs_dataset = create_control_prs_dataset(
                    biobank=biobank,
                    analysis_id=control_analysis_id,
                    phenotype=phenotype,
                    selected_pgs=selected_pgs,
                    pcs_source=args.pcs_source,
                    ancestry_source=args.ancestry_source,
                )
            except FileNotFoundError as exc:
                print(
                    f"> Skipping control PRSDataset for {control_analysis_id} "
                    f"({biobank}) because a required source file is unavailable: "
                    f"{exc}"
                )
                continue

            out_dir = f"data/harmonized_data/{control_analysis_id}/{biobank}/"
            makedir(out_dir)

            # The full dataset is not fold-specific, so keep one copy here.
            prs_dataset.save(osp.join(out_dir, "full_data.pkl"))

            if args.match_reference_split:
                fold_splits = (
                    (
                        fold,
                        *split_like_reference_dataset(
                            prs_dataset=prs_dataset,
                            reference_analysis_id=analysis_id,
                            biobank=biobank,
                            fold=fold,
                        ),
                    )
                    for fold in _fold_names(args.n_folds)
                )
            else:
                fold_splits = (
                    (f"fold_{fold_idx}", train_data, test_data)
                    for fold_idx, (train_data, test_data) in enumerate(
                        prs_dataset.k_fold_split(
                            n_splits=args.n_folds,
                            random_state=args.seed + i,
                        ),
                        start=1,
                    )
                )

            for fold, train_data, test_data in fold_splits:
                fold_dir = osp.join(out_dir, fold)
                makedir(fold_dir)
                train_data.save(osp.join(fold_dir, "train_data.pkl"))
                test_data.save(osp.join(fold_dir, "test_data.pkl"))

                split_source = (
                    f"matched from {analysis_id}"
                    if args.match_reference_split
                    else "created"
                )
                print(
                    f"> {fold} {split_source} ({biobank}): "
                    f"{train_data.N} train / {test_data.N} test samples."
                )

        for pgs in selected_pgs:
            row = template_row.copy()
            row["AnalysisID"] = control_analysis_id
            row["PGS"] = pgs
            row[args.disease_flag_col] = bool(pgs == disease_pgs)
            row["Testing_R2"] = sampled_r2_map.get(pgs, np.nan)
            row["Testing_R2_Metric"] = sampled_metric_map.get(pgs, "R2")
            row["Sampled_As_Noninformative"] = bool(sampled_flag_map.get(pgs, False))
            row["Testing_Biobank"] = args.biobank
            row["Testing_R2_Threshold"] = float(args.r2_threshold)
            if "PGSCatalog_ID" in row:
                row["PGSCatalog_ID"] = pgs
            if "PGS_Name" in row:
                row["PGS_Name"] = pgs_name_map.get(pgs, pgs)
            if "Notes" in row:
                if pgs == disease_pgs:
                    row["Notes"] = (
                        "Disease-specific score retained in control analysis "
                        f"(Testing_R2={row['Testing_R2']:.6g}, "
                        f"metric={row['Testing_R2_Metric']})"
                    )
                else:
                    row["Notes"] = (
                        f"Sampled non-informative control score "
                        f"(biobank={args.biobank}, "
                        f"{row['Testing_R2_Metric']}<{args.r2_threshold}, "
                        f"Testing_R2={row['Testing_R2']:.6g})"
                    )

            control_rows.append(row)

    if len(control_rows) == 0:
        raise ValueError("No control rows were generated.")

    control_df = pd.DataFrame(control_rows)

    # Keep the same column order as the input analysis table.
    ordered_cols = [c for c in analysis_df.columns if c in control_df.columns]
    extra_cols = [c for c in control_df.columns if c not in ordered_cols]
    control_df = control_df[ordered_cols + extra_cols]

    output_dir = osp.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    control_df.to_csv(args.output_file, index=False)
    print(
        f"Saved {len(control_df)} control rows for {len(requested_analysis_ids)} analyses "
        f"to {args.output_file}"
    )
