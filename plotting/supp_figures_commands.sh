#!/bin/bash

source env/moe/bin/activate

# ----------------- Section 1: Sex-differentiated phenotypes -----------------

echo "------------------------------------------------"
echo "> Sex-differentiated Phenotypes\n\n"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_1/" \
    --grid "accuracy_subpanels_all_cartagene.pdf\\nmixing_weights_categorical_all_cartagene.png" \
    --output "figures/supplementary_figures/sex_differentiated_accuracy_mixing_weights_cag.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/gate_parameters/" \
    --grid "WHR_SEX_UKB.eps,LOG_TST_SEX_UKB.eps,LOG_CRTN_SEX_UKB.eps,URT_SEX_UKB.eps\\nWHR_SEX_CaG.eps,,LOG_CRTN_SEX_CaG.eps,URT_SEX_CaG.eps" \
    --output "figures/supplementary_figures/sex_differentiated_gate_params.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_1/" \
    --grid "mixing_weights_by_sex_urate_ukbb.png,mixing_weights_by_ancestry_urate_ukbb.png\\nmixing_weights_by_sex_urate_cartagene.png,mixing_weights_by_ancestry_urate_cartagene.png" \
    --output "figures/supplementary_figures/sex_differentiated_mixing_weight_URT.pdf"


python plotting/make_fig_grid.py \
    --fig-dir "figures/section_1/" \
    --grid "accuracy_stratified_by_ancestry_whr_ukbb.pdf,accuracy_stratified_by_ancestry_whr_cartagene.pdf\\naccuracy_stratified_by_ancestry_testosterone_ukbb.pdf,\\naccuracy_stratified_by_ancestry_creatinine_ukbb.pdf,accuracy_stratified_by_ancestry_creatinine_cartagene.pdf" \
    --output "figures/supplementary_figures/sex_differentiated_stratified_accuracy_wo_URT.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_1/" \
    --grid "accuracy_stratified_by_ancestry_urate_ukbb.pdf,accuracy_stratified_by_ancestry_urate_cartagene.pdf\\naccuracy_stratified_by_ancestry_female_age_urate_ukbb.pdf,accuracy_stratified_by_ancestry_female_age_urate_cartagene.pdf" \
    --output "figures/supplementary_figures/sex_differentiated_stratified_accuracy_URT.pdf"


python plotting/make_fig_grid.py \
    --fig-dir "figures/calpred/" \
    --grid "WHR_SEX_ukbb.eps,WHR_SEX_cartagene.eps\\nLOG_CRTN_SEX_ukbb.eps,LOG_CRTN_SEX_cartagene.eps\\nURT_SEX_ukbb.eps,URT_SEX_cartagene.eps" \
    --output "figures/supplementary_figures/sex_differentiated_calpred_results.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_1/" \
    --grid "accuracy_subpanels_non_eur_all_ukbb.pdf\\naccuracy_subpanels_non_eur_all_cartagene.pdf" \
    --output "figures/supplementary_figures/sex_differentiated_accuracy_non_EUR.pdf"


python plotting/make_fig_grid.py \
    --fig-dir "figures/section_1/" \
    --grid "phenotypic_variance_WHR_SEX_ukbb.pdf,phenotypic_variance_WHR_SEX_cartagene.pdf\\nphenotypic_variance_LOG_CRTN_SEX_ukbb.pdf,phenotypic_variance_LOG_CRTN_SEX_cartagene.pdf\\nphenotypic_variance_URT_SEX_ukbb.pdf,phenotypic_variance_URT_SEX_cartagene.pdf" \
    --output "figures/supplementary_figures/sex_differentiated_phenotype_variance.pdf"

# ----------------- Section 2: Ancestry-stratified analysis -----------------

echo "------------------------------------------------"
echo "> Ancestry-stratified Analysis\n\n"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_2/" \
    --grid "accuracy_metrics_all_cartagene.pdf\\nmixing_weight_similarity_cosine_all_cartagene.pdf" \
    --output "figures/supplementary_figures/ancestry_stratified_accuracy_concordance_cag.pdf"

# First, plot the gate params:
python plotting/make_fig_grid.py \
    --fig-dir "figures/gate_parameters/" \
    --grid "HEIGHT_MA_UKB.eps,HEIGHT_MA_CaG.eps\\nLOG_BMI_MA_UKB.eps,LOG_BMI_MA_CaG.eps" \
    --output "figures/supplementary_figures/ancestry_stratified_HT_BMI_gate_params.pdf"


python plotting/make_fig_grid.py \
    --fig-dir "figures/gate_parameters/" \
    --grid "DBP_MA_UKB.eps,DBP_MA_CaG.eps\\nSBP_MA_UKB.eps,SBP_MA_CaG.eps" \
    --output "figures/supplementary_figures/ancestry_stratified_BP_gate_params.pdf"


python plotting/make_fig_grid.py \
    --fig-dir "figures/gate_parameters/" \
    --grid "ASTHMA_MA_UKB.eps,ASTHMA_MA_CaG.eps\\nT2D_MA_UKB.eps,T2D_MA_CaG.eps" \
    --output "figures/supplementary_figures/ancestry_stratified_disease_gate_params.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/gate_parameters/" \
    --grid "LOG_TG_MA_UKB.eps,LOG_TG_MA_CaG.eps\\nLOG_HDL_MA_UKB.eps,LOG_HDL_MA_CaG.eps" \
    --output "figures/supplementary_figures/ancestry_stratified_TG_HDL_gate_params.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/gate_parameters/" \
    --grid "LDL_MA_UKB.eps,LDL_MA_CaG.eps\\nTC_MA_UKB.eps,TC_MA_CaG.eps" \
    --output "figures/supplementary_figures/ancestry_stratified_LDL_TC_gate_params.pdf"

# Second, plot the calpred params (for a subset of the phenotypes):

python plotting/make_fig_grid.py \
    --fig-dir "figures/calpred/" \
    --grid "LOG_BMI_MA_ukbb.eps,LOG_BMI_MA_cartagene.eps\\nLDL_MA_ukbb.eps,LDL_MA_cartagene.eps\\nTC_MA_ukbb.eps,TC_MA_cartagene.eps" \
    --output "figures/supplementary_figures/ancestry_stratified_calpred_results.pdf"

# Third, plot the PRS mixture graphs:

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_2/" \
    --grid "mixture_graphs_HEIGHT_MA_ukbb.png\\nmixture_graphs_HEIGHT_MA_cartagene.png\\nmixture_graphs_LOG_BMI_MA_ukbb.png\\nmixture_graphs_LOG_BMI_MA_cartagene.png" \
    --output "figures/supplementary_figures/ancestry_stratified_HT_BMI_mixture_graphs.pdf"


python plotting/make_fig_grid.py \
    --fig-dir "figures/section_2/" \
    --grid "mixture_graphs_DBP_MA_ukbb.png\\nmixture_graphs_DBP_MA_cartagene.png\\nmixture_graphs_SBP_MA_ukbb.png\\nmixture_graphs_SBP_MA_cartagene.png" \
    --output "figures/supplementary_figures/ancestry_stratified_BP_mixture_graphs.pdf"


python plotting/make_fig_grid.py \
    --fig-dir "figures/section_2/" \
    --grid "mixture_graphs_ASTHMA_MA_ukbb.png\\nmixture_graphs_ASTHMA_MA_cartagene.png\\nmixture_graphs_T2D_MA_ukbb.png\\nmixture_graphs_T2D_MA_cartagene.png" \
    --output "figures/supplementary_figures/ancestry_stratified_disease_mixture_graphs.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_2/" \
    --grid "mixture_graphs_LOG_TG_MA_ukbb.png\\nmixture_graphs_LOG_TG_MA_cartagene.png\\nmixture_graphs_LOG_HDL_MA_ukbb.png\\nmixture_graphs_LOG_HDL_MA_cartagene.png" \
    --output "figures/supplementary_figures/ancestry_stratified_TG_HDL_mixture_graphs.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_2/" \
    --grid "mixture_graphs_LDL_MA_ukbb.png\\nmixture_graphs_LDL_MA_cartagene.png\\nmixture_graphs_TC_MA_ukbb.png\\nmixture_graphs_TC_MA_cartagene.png" \
    --output "figures/supplementary_figures/ancestry_stratified_LDL_TC_mixture_graphs.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_2/" \
    --grid "admixture_graphs_MID_ukbb.png\\nadmixture_graphs_MID_cartagene.png" \
    --output "figures/supplementary_figures/ancestry_stratified_HT_mid_mixture_graphs.pdf"


# ----------------- Section 3: Medication-use -----------------

echo "------------------------------------------------"
echo "> Medication-use\n\n"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_3/" \
    --grid "mixing_weights_by_age_sex_LDL_MA_cartagene.png,accuracy_medication_use_LDL_MA_cartagene.pdf,mixing_weights_by_age_sex_LDL_ADJ_MA_cartagene.png,accuracy_stratified_LDL_ADJ_MA_cartagene.pdf\\nmixing_weights_by_age_sex_TC_MA_cartagene.png,accuracy_medication_use_TC_MA_cartagene.pdf,mixing_weights_by_age_sex_TC_ADJ_MA_cartagene.png,accuracy_stratified_TC_ADJ_MA_cartagene.pdf" \
    --output "figures/supplementary_figures/medication_use_cholesterol_stratified_accuracy_cag.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_3/" \
    --grid "mixing_weights_by_age_sex_DBP_MA_cartagene.png,accuracy_medication_use_DBP_MA_cartagene.pdf,mixing_weights_by_age_sex_DBP_ADJ_MA_cartagene.png,accuracy_stratified_DBP_ADJ_MA_cartagene.pdf\\nmixing_weights_by_age_sex_SBP_MA_cartagene.png,accuracy_medication_use_SBP_MA_cartagene.pdf,mixing_weights_by_age_sex_SBP_ADJ_MA_cartagene.png,accuracy_stratified_SBP_ADJ_MA_cartagene.pdf" \
    --output "figures/supplementary_figures/medication_use_BP_stratified_accuracy_cag.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_3/" \
    --grid "mixing_weights_by_age_sex_DBP_MA_ukbb.png,accuracy_medication_use_DBP_MA_ukbb.pdf,mixing_weights_by_age_sex_DBP_ADJ_MA_ukbb.png,accuracy_stratified_DBP_ADJ_MA_ukbb.pdf\\nmixing_weights_by_age_sex_SBP_MA_ukbb.png,accuracy_medication_use_SBP_MA_ukbb.pdf,mixing_weights_by_age_sex_SBP_ADJ_MA_ukbb.png,accuracy_stratified_SBP_ADJ_MA_ukbb.pdf" \
    --output "figures/supplementary_figures/medication_use_BP_stratified_accuracy_ukbb.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_3/" \
    --grid "mixing_weights_by_age_sex_TC_MA_ukbb.png,accuracy_medication_use_TC_MA_ukbb.pdf,mixing_weights_by_age_sex_TC_ADJ_MA_ukbb.png,accuracy_stratified_TC_ADJ_MA_ukbb.pdf" \
    --output "figures/supplementary_figures/medication_use_TC_stratified_accuracy_ukbb.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_3/" \
    --grid "ancestry_classifier_concordance_all_cartagene.pdf,\\nancestry_classifier_concordance_med_adj_phenotypes_cartagene.pdf,accuracy_metrics_med_adj_all_cartagene.pdf" \
    --output "figures/supplementary_figures/medication_use_accuracy_concordance_cag.pdf"

# ----------------- Section 4: Cross-trait analysis -----------------

echo "------------------------------------------------"
echo "> Cross-trait Analysis\n\n"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_4/" \
    --grid "mixture_graphs_T2D_MT_ukbb.png,mixture_graphs_T1D_MT_ukbb.png,mixture_graphs_ASTHMA_MT_ukbb.png\\nmixture_graphs_GOUT_MT_ukbb.png,mixture_graphs_CAD_MT_ukbb.png,mixture_graphs_AF_MT_ukbb.png\\nmixture_graphs_HTN_MT_ukbb.png,mixture_graphs_STR_MT_ukbb.png,mixture_graphs_HF_MT_ukbb.png" \
    --output "figures/supplementary_figures/cross_trait_mixture_graphs.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_4/" \
    --grid "mixture_graphs_T2D_MT_CTRL_ukbb.png,mixture_graphs_T1D_MT_CTRL_ukbb.png,mixture_graphs_ASTHMA_MT_CTRL_ukbb.png\\nmixture_graphs_GOUT_MT_CTRL_ukbb.png,mixture_graphs_CAD_MT_CTRL_ukbb.png,mixture_graphs_AF_MT_CTRL_ukbb.png\\nmixture_graphs_HTN_MT_CTRL_ukbb.png,mixture_graphs_STR_MT_CTRL_ukbb.png,mixture_graphs_HF_MT_CTRL_ukbb.png" \
    --output "figures/supplementary_figures/cross_trait_mixture_graphs_ctrl.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_4/" \
    --grid "accuracy_metrics_control_ukbb.pdf\\nmixing_weight_disease_prs_control_phenotypes_ukbb.pdf" \
    --output "figures/supplementary_figures/cross_trait_ctrl_analysis.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_4/" \
    --grid "accuracy_disease_prs_age_sex_T2D_MT_ukbb.png,accuracy_disease_prs_age_sex_T1D_MT_ukbb.png\\naccuracy_disease_prs_age_sex_HTN_MT_ukbb.png,accuracy_disease_prs_age_sex_CAD_MT_ukbb.png" \
    --output "figures/supplementary_figures/cross_trait_stratified_accuracy_per_disease_ukbb.pdf"

cartagene_t1d_panel=""
if [[ -f "figures/section_4/accuracy_disease_prs_age_sex_T1D_MT_cartagene.png" ]]; then
    cartagene_t1d_panel="accuracy_disease_prs_age_sex_T1D_MT_cartagene.png"
else
    echo "Warning: CARTaGENE T1D age/sex accuracy panel is unavailable; leaving its grid cell blank." >&2
fi

python plotting/make_fig_grid.py \
    --fig-dir "figures/section_4/" \
    --grid "accuracy_disease_prs_age_sex_T2D_MT_cartagene.png,${cartagene_t1d_panel}\\naccuracy_disease_prs_age_sex_HTN_MT_cartagene.png,accuracy_disease_prs_age_sex_CAD_MT_cartagene.png" \
    --output "figures/supplementary_figures/cross_trait_stratified_accuracy_per_disease_cartagene.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/gate_parameters/" \
    --grid "T2D_MT_UKB.eps,T1D_MT_UKB.eps\\nHTN_MT_UKB.eps,CAD_MT_UKB.eps" \
    --output "figures/supplementary_figures/cross_trait_gate_params.pdf"

python plotting/make_fig_grid.py \
    --fig-dir "figures/calpred/" \
    --grid "T2D_MT_ukbb.eps,T2D_MT_cartagene.eps\\nT1D_MT_ukbb.eps,T1D_MT_cartagene.eps\\nHTN_MT_ukbb.eps,HTN_MT_cartagene.eps\\nCAD_MT_ukbb.eps,CAD_MT_cartagene.eps" \
    --output "figures/supplementary_figures/cross_trait_calpred_results.pdf"

cp figures/section_4/mixing_weight_quartile_metric_panels_all_phenotypes_ukbb.pdf figures/supplementary_figures/cross_trait_mixing_weight_quartile_metric_panels_ukbb.pdf
