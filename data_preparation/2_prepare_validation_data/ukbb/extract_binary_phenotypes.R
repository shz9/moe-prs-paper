#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(bigreadr)
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(PheWAS)
})

##########################
# CLI options
##########################

option_list <- list(
  make_option(c("-f","--ukb-file"),  type="character", default=NULL,
              help="Path to UKB wide-format CSV"),
  make_option(c("--withdrawn-file"),  type="character", default=NULL,
              help="Path to the file with the IDs of withdrawn individuals."),
  make_option(c("-p","--phecodes"), type="character", default=NULL,
              help="Comma-separated list of phecodes"),
  make_option(c("-n","--phenotype-names"), type="character", default=NULL,
              help="Comma-separated names for phenotypes (same order as --phecodes). Repeated phenotype names are allowed."),
  make_option(c("-o","--outdir"),   type="character", default="data/ukbb-selected-phecodes",
              help="Output directory"),
  make_option(c("--include-cancer"),       action="store_true", default=FALSE,
              help="Include cancer-related ICD fields"),
  make_option(c("--include-selfreported"), action="store_true", default=FALSE,
              help="Include self-reported fields (20002 + coding609)"),
  make_option(c("--apply-phecode-exclusion"), action="store_true", default=FALSE,
              help="Apply phecode exclusions")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$`ukb-file`) || is.null(opt$phecodes)) {
  stop("Usage: Rscript prepare-selected-phecodes-cli.R -f ukb.csv -p 411.2,274 [--include-cancer] [--include-selfreported]")
}

ukb_csv <- opt$`ukb-file`
phecodes_requested <- unlist(strsplit(opt$phecodes, ",\\s*"))
out_dir <- opt$outdir
include_cancer <- opt$`include-cancer`
include_selfreported <- opt$`include-selfreported`
apply_phecode_exclusions <- opt$`apply-phecode-exclusion`
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

##########################
# Parse phenotype names
##########################

if (!is.null(opt$`phenotype-names`)) {
  phenotype_names <- unlist(strsplit(opt$`phenotype-names`, ",\\s*"))
  if (length(phenotype_names) != length(phecodes_requested)) {
    stop("--phenotype-names must have same length as --phecodes")
  }
} else {
  phenotype_names <- phecodes_requested
}

# Build mapping from phenotype name -> all associated phecodes
phenotype_map <- tibble(
  phecode = phecodes_requested,
  phenotype = phenotype_names
) %>%
  group_by(phenotype) %>%
  summarise(
    phecodes = list(unique(phecode)),
    .groups = "drop"
  )

message("phecode_map rows: ", nrow(phecode_map_icd10),
        " | vocabs: ", paste(unique(phecode_map_icd10$vocabulary_id), collapse=", "))

##########################
# Read UKB fields
##########################

if (!file.exists(ukb_csv)) stop("UKB CSV not found: ", ukb_csv)
sex <- fread2(ukb_csv, select = "22001-0.0")[[1]]
eid <- fread2(ukb_csv, select = "eid")[[1]]

icd10_select <- c(
  # Causes of death
  paste0("40001-", 0:1, ".0"),      # underlying cause
  paste0("40002-0.", 0:13),         # contributory causes
  paste0("40002-1.", 0:13),

  # Hospital inpatient: primary, secondary, external causes
  paste0("41201-0.", 0:21),         # external causes
  paste0("41202-0.", 0:79),         # primary ICD10 diagnoses
  paste0("41204-0.", 0:209),        # secondary ICD10 diagnoses

  # Summary diagnoses
  paste0("41270-0.", 0:258)         # summary ICD10 diagnoses
)

if (include_cancer) {
  icd10_select <- unique(c(icd10_select, paste0("40006-", 0:16, ".0")))
}

df_ICD10 <- fread2(ukb_csv, colClasses = "character", select = icd10_select)

if (include_selfreported) {
  # NOTE: Download from: https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id=609
  coding609 <- fread2("data/phewas/coding609.tsv")
  sr_cols <- c(paste0("20002-0.", 0:33), paste0("20002-1.", 0:33),
               paste0("20002-2.", 0:33), paste0("20002-3.", 0:33))
  df_sr <- fread2(ukb_csv, colClasses = "character", select = sr_cols)

  df_sr <- df_sr %>%
    mutate_all(~ as.character(factor(., levels = coding609$coding,
                                     labels = coding609$meaning)))

  df_ICD10 <- bind_cols(df_ICD10, df_sr)
}

# NOTE: Excluding ICD9 codes from the analysis for now.
# In future, we may need to do the mapping using: phecode_icd9_rolled.csv

##########################
# Wide -> long tibbles
##########################

build_long <- function(df, vocab_id) {
  df %>%
    mutate_all(~ ifelse(. == "", NA, .)) %>%
    mutate(id = row_number()) %>%
    pivot_longer(-id, values_to = "code", values_drop_na = TRUE) %>%
    group_by(id, code) %>%
    summarise(count = n(), .groups = "drop") %>%
    ungroup() %>%
    transmute(
      id            = as.integer(id),
      vocabulary_id = vocab_id,
      code          = trimws(as.character(code)),
      count         = as.integer(count)
    ) %>%
    as_tibble()
}

# vocabulary_id values must match exactly what is in phecode_map
id_icd10_count <- build_long(df_ICD10, "ICD10")

codes_tab <- id_icd10_count

message("codes_tab rows: ", nrow(codes_tab),
        " | vocabs: ", paste(unique(codes_tab$vocabulary_id), collapse=", "))

##########################
# createPhenotypes
##########################

phen_wide <- createPhenotypes(
  id_icd10_count,
  id.sex                 = tibble(id = seq_along(sex), sex = c("F","M")[sex + 1L]),
  vocabulary.map         = mutate_at(phecode_map_icd10, "code", ~ sub("\\.", "", .)),
  min.code.count         = 1,
  add.phecode.exclusions = apply_phecode_exclusions,
  full.population.ids    = seq_along(sex),
  translate              = TRUE
)

phen_wide <- phen_wide[order(phen_wide$id), , drop = FALSE]

##########################
# Select requested phecodes
##########################

present <- intersect(phecodes_requested, colnames(phen_wide))
missing <- setdiff(phecodes_requested, present)
if (length(missing) > 0) warning("Missing phecodes: ", paste(missing, collapse = ", "))

phen_selected <- phen_wide %>%
  select(all_of(c("id", present)))

phen_selected$eid <- eid

##########################
# Exclude withdrawn samples
##########################

if (!is.null(opt$`withdrawn-file`)) {
  withdrawn_df <- fread2(opt$`withdrawn-file`)

  message("> Removed data for withdrawn samples.")
  message("> Sample size before: ", nrow(phen_selected))
  phen_selected <- phen_selected %>%
    filter(!eid %in% withdrawn_df$V1)
  message("> Sample size after: ", nrow(phen_selected))
}

##########################
# Helper: OR across multiple phecodes
##########################

collapse_phecodes_or <- function(df, phecode_cols) {
  if (length(phecode_cols) == 0) {
    return(rep(NA_integer_, nrow(df)))
  }

  x <- df %>%
    select(all_of(phecode_cols)) %>%
    mutate(across(everything(), ~ suppressWarnings(as.numeric(.))))

  x_mat <- as.matrix(x)

  # True if any associated phecode is 1
  any_case <- apply(x_mat == 1, 1, function(z) any(z, na.rm = TRUE))

  # True only if every associated phecode is NA
  all_na <- apply(is.na(x_mat), 1, all)

  out <- ifelse(any_case, 1L, ifelse(all_na, NA_integer_, 0L))
  as.integer(out)
}

##########################
# Write one file per phenotype name
##########################

for (i in seq_len(nrow(phenotype_map))) {

  fname <- phenotype_map$phenotype[i]
  pcs <- phenotype_map$phecodes[[i]]

  present_pcs <- intersect(pcs, names(phen_selected))
  missing_pcs <- setdiff(pcs, names(phen_selected))

  if (length(missing_pcs) > 0) {
    warning("Phenotype '", fname, "' is missing phecodes: ", paste(missing_pcs, collapse = ", "))
  }

  if (length(present_pcs) == 0) next

  pheno <- collapse_phecodes_or(phen_selected, present_pcs)

  out_df <- tibble(
    FID = phen_selected$eid,
    IID = phen_selected$eid,
    PHENO = pheno
  ) %>%
    filter(!is.na(PHENO))

  safe_fname <- gsub("[[:space:]]+", "_", fname)
  safe_fname <- gsub("[^A-Za-z0-9_\\-\\.]", "", safe_fname)

  write.table(
    out_df,
    file = file.path(out_dir, paste0(safe_fname, ".txt")),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
  )
}

message("Done. Output written to: ", out_dir)
