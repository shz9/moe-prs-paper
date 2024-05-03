import os.path as osp
import pandas as pd
import numpy as np
import glob
import hail as hl
import onnxruntime as rt
import argparse
from magenpy.utils.system_utils import makedir


def project_samples_hail(bed_files,
                         pc_loadings_path,
                         genotype_ref_genome='GRCh38',
                         loadings_ref_genome='GRCh38'):
    """
    Project the samples onto the PC loadings using Hail. This function takes a list of BED files
    containing the genotypes of the samples, the path to the file containing the PC loadings, and
    the reference genome build of the genotypes and the PC loadings. The function returns a pandas
    DataFrame containing the projected PCs for the samples.

    """

    assert genotype_ref_genome in ('GRCh37', 'GRCh38')
    assert loadings_ref_genome in ('GRCh37', 'GRCh38')

    loadings_ht = hl.read_table(pc_loadings_path)

    if genotype_ref_genome != loadings_ref_genome:
        # If the reference genome of the genotypes is different from the reference genome of the PC loadings,
        # we need to liftover the coordinates of the PC loadings to the reference genome of the genotypes.

        assert args.liftover_chain is not None, "A liftover chain file is required to lift over the coordinates."

        rg37 = hl.get_reference('GRCh37')
        rg38 = hl.get_reference('GRCh38')

        if genotype_ref_genome == 'GRCh37':
            rg38.add_liftover(args.liftover_chain, rg37)
            loadings_ht = loadings_ht.annotate(new_locus=hl.liftover(loadings_ht.locus, 'GRCh37'))
            loadings_ht = loadings_ht.filter(hl.is_defined(loadings_ht.new_locus))
        else:
            rg37.add_liftover(args.liftover_chain, rg38)
            loadings_ht = loadings_ht.annotate(new_locus=hl.liftover(loadings_ht.locus, 'GRCh38'))
            loadings_ht = loadings_ht.filter(hl.is_defined(loadings_ht.new_locus))

        # Update the row_key:
        loadings_ht = loadings_ht.key_by(locus=loadings_ht.new_locus, alleles=loadings_ht.alleles)

    # Loop over the BED files and combine them into one matrix table:
    combined = None
    for f in glob.glob(bed_files):
        mt = hl.import_plink(bed=f,
                             bim=f.replace(".bed", ".bim"),
                             fam=f.replace(".bed", ".fam"),
                             reference_genome=genotype_ref_genome)
        if combined is None:
            combined = mt
        else:
            combined = combined.union_rows(mt)

    # Annotated the combined matrix table with the PC loadings and allele frequencies:
    mt = combined.annotate_rows(
        pca_loadings=loadings_ht[combined.row_key]['loadings'],
        pca_af=loadings_ht[combined.row_key]['pca_af']
    )

    ht = hl.experimental.pc_project(mt.GT, loadings_ht.loadings, loadings_ht.pca_af)
    df = ht.to_pandas()

    # Obtain the PC scores:
    pcs = np.array(list(map(np.array, list(df.scores.values))))

    # Create a DataFrame containing the PC scores + sample IDs:
    pc_df = pd.DataFrame(pcs, columns=[f'PC_{i}' for i in range(1, pcs.shape[1] + 1)])
    pc_df['IID'] = mt.cols().s.collect()
    pc_df['FID'] = mt.cols().fam_id.collect()

    return pc_df


def assign_ancestry_random_forest(projected_pcs, rf_model_path, min_prob=0.5):
    """
    Assign ancestry to a set of samples using a pretrained Random Forest model
    from gnomad v3.1. The ancestry is assigned based on the maximum probability
    from the random forest model. We only assign ancestry to samples whose highest
    probability exceeds the threshold `min_prob`. By default, according to the gnomad
    documentation, `min_prob` is set to 0.5. More details can be found here:

    https://github.com/broadinstitute/gnomad_qc/blob/0008e32ab0971ce31dd91318f57af9482fbabcdf/gnomad_qc/example_notebooks/ancestry_classification_using_gnomad_rf.ipynb

    """

    # Load the ONNX model
    sess = rt.InferenceSession(rf_model_path)

    # Get the input name for the ONNX model
    input_name = sess.get_inputs()[0].name

    # Run the model on your test data
    clsf, probs = sess.run(None, {input_name: projected_pcs.astype(np.float32)})
    probs = pd.DataFrame(probs)

    # Assign the ancestry with the maximum probability
    probs['ancestry'] = clsf

    # If the maximum probability is less than min_prob, assign the sample as "other"
    probs['ancestry'] = np.where(probs.drop(columns='ancestry').max(axis=1) < min_prob, 'oth', probs['ancestry'])

    return probs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Assign ancestry to a set of samples using '
                                                 'PC loadings and Random Forest model from gnomad v3.1.')

    parser.add_argument('--bed-files', dest='bed_files', type=str, required=True,
                        help='The path to the BED files containing the genotypes of the samples.')
    parser.add_argument('--genotype-ref-genome', dest='genotype_ref_genome',
                        type=str, default='GRCh38',
                        choices={'GRCh37', 'GRCh38'},
                        help='The genome build used to generate the genotypes in the BED files.')
    parser.add_argument('--pca-loadings', dest='pca_loadings', type=str, required=True,
                        help='The path to the file containing the PC loadings from gnomad v3.1.')
    parser.add_argument('--pca-ref-genome', dest='pca_ref_genome',
                        type=str, default='GRCh38',
                        choices={'GRCh37', 'GRCh38'},
                        help='The genome build used for the samples from which PC loadings are derived.')
    parser.add_argument('--liftover-chain', dest='liftover_chain', type=str)
    parser.add_argument('--output-dir', dest='output_dir', type=str, required=True,
                        help='The path to the output directory.')
    parser.add_argument('--rf-model', dest='rf_model', type=str, required=True,
                        help='The path to the Random Forest model stored in ONNX format from gnomad v3.1.')

    args = parser.parse_args()

    hl.init(spark_conf={'spark.driver.memory': '50g', 'spark.executor.memory': '50g'})

    # Project the samples onto the PC loadings
    sample_pcs = project_samples_hail(args.bed_files,
                                      args.pca_loadings,
                                      args.genotype_ref_genome,
                                      args.pca_ref_genome)

    # Save the projected PCs to file
    makedir(args.output_dir)
    sample_pcs.to_csv(osp.join(args.output_dir, f"projected_sample_pcs.txt"),
                      sep="\t", index=False)

    # --------------------------------------------------------------------------------------------
    # Clustering and generating population labels:

    # Assign ancestry to the samples using the Random Forest model
    # The RF model only uses the first 16 PCs for ancestry assignment:
    ancestry_probs = assign_ancestry_random_forest(sample_pcs[[f'PC_{i}' for i in range(1, 17)]].values,
                                                   args.rf_model)
    ancestry_probs[['FID', 'IID']] = sample_pcs[['FID', 'IID']]

    # Print the inferred ancestry distribution:
    print("Inferred ancestry distribution:")
    print(ancestry_probs.groupby('ancestry').size())

    # Save the ancestry probabilities to file
    ancestry_probs.to_csv(osp.join(args.output_dir, "ancestry_assignments.txt"), sep="\t", index=False)
