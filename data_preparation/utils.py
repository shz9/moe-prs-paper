import pandas as pd


def liftover_coordinates(dataframe,
                         chr_col='CHR',
                         pos_col='POS',
                         source='hg19',
                         target='hg38',
                         chain_file=None):
    """
    Lift over the variant coordinates in a dataframe from one genome build to another.

    :param dataframe: A pandas DataFrame containing the variant information.
    :param chr_col: The name of the column containing the chromosome information.
    :param pos_col: The name of the column containing the position information.
    :param source: The source genome build (default: 'hg19').
    :param target: The target genome build (default: 'hg38').
    :param chain_file: The path to the chain file to use for the liftover (default: None).
    """

    assert chr_col in dataframe.columns
    assert pos_col in dataframe.columns

    if chain_file is None:
        from liftover import get_lifter
        converter = get_lifter(source, target)
    else:
        from liftover import ChainFile
        converter = ChainFile(chain_file, source, target)

    def convert_coords(x):
        try:
            res = converter[x[chr_col]][x[pos_col]][0]
            chrom = int(res[0].replace('chr', ''))
            pos = int(res[1])
            return chrom, pos
        except Exception:
            return -1, -1

    return zip(*dataframe.apply(convert_coords, axis=1))
