import argparse
import hail as hl

NO_CHR_TO_CHR_CONTIG_RECODING = {
    "1": "chr1",
    "2": "chr2",
    "3": "chr3",
    "4": "chr4",
    "5": "chr5",
    "6": "chr6",
    "7": "chr7",
    "8": "chr8",
    "9": "chr9",
    "10": "chr10",
    "11": "chr11",
    "12": "chr12",
    "13": "chr13",
    "14": "chr14",
    "15": "chr15",
    "16": "chr16",
    "17": "chr17",
    "18": "chr18",
    "19": "chr19",
    "20": "chr20",
    "21": "chr21",
    "22": "chr22",
    "X": "chrX",
    "Y": "chrY",
    "MT": "chrM",
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Annotate Hail table with dbSNP rsids')
    parser.add_argument('--input', dest='input', type=str, help='Input Hail table',
                        default='data/gnomad_data/gnomad.v3.1.pca_loadings.ht')
    parser.add_argument('--output-file', dest='output_file', type=str, help='Output file',
                        default='data/gnomad_data/gnomad.v3.1.pca_loadings_annotated.tsv.gz')
    parser.add_argument('--dbsnp', dest='dbsnp', type=str, help='dbSNP VCF file',
                        default='/home/szabad/projects/ctb-sgravel/data/genome_annotations'
                                '/GRCh38/dbSNP/VCF/All_20180418.vcf.gz')

    args = parser.parse_args()

    ht = hl.read_table(args.input)

    dbSNP = hl.import_vcf(
        args.dbsnp, force=True,
        reference_genome='GRCh38',
        contig_recoding=NO_CHR_TO_CHR_CONTIG_RECODING
    )

    dbSNP_rows = dbSNP.rows()

    ht = ht.annotate(rsid=dbSNP_rows[ht.key].rsid)
    pandas_tab = ht.to_pandas()

    pandas_tab.to_csv(args.output_file, sep="\t", index=False)
