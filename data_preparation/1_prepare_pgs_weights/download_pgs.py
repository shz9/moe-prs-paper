import json
import argparse
import os.path as osp
import urllib.request
import pandas as pd
from tqdm import tqdm
from magenpy.utils.system_utils import makedir


def download_pgs_from_catalog(pgs_id, data_dir=None, genome_build='GRCh37'):
    """
    Download polygenic score file from the PGS catalog.
    For more information about the available polygenic scores and their 
    associated metadata, consult the official website of the PGS catalog:
    
    https://www.pgscatalog.org/
    
    :param pgs_id: A string corresponding to the ID of the polygenic score (e.g. "PGS000001")
    :param data_dir: A string corresponding to the directory where the PGS file will be downloaded.
    :param genome_build: The genome build where the variant coordinates are derived from
    """
    
    pgs_info = json.loads(urllib.request.urlopen(f"https://www.pgscatalog.org/rest/score/{pgs_id}").read())
    
    # If the genome build for the variants is specified as `genome_build`, download the default scoring file.
    # Otherwise, download the harmonized scoring file from the PGS catalog.

    if pgs_info["variants_genomebuild"] == genome_build:
        ftp_link = pgs_info["ftp_scoring_file"]
    else:
        ftp_link = pgs_info["ftp_harmonized_scoring_files"][genome_build]["positions"]

    print(ftp_link)

    if data_dir is None:
        return pd.read_csv(ftp_link, comment='#', sep="\t")
    else:
        f_name = osp.join(data_dir, genome_build, f"{pgs_id}.txt.gz")
        makedir(osp.dirname(f_name))
        urllib.request.urlretrieve(ftp_link, f_name)
    

def download_trait_pgs_from_catalog(trait_id, data_dir=None, genome_build='GRCh37'):
    """
    Download polygenic score files corresponding to a particular trait_id from the PGS catalog.
    For more information about the available traits, associated polygenic scores, and their metadata,
    consult the official website of the PGS catalog:
    
    https://www.pgscatalog.org/
    
    :param trait_id: A string corresponding to the ID of the trait (e.g. "EFO_0000305")
    :param data_dir: A string corresponding to the directory where the PGS files will be downloaded.
    :param genome_build: The genome build where the variant coordinates are derived from
    """
    
    trait_info = json.loads(urllib.request.urlopen(f"https://www.pgscatalog.org/rest/trait/{trait_id}").read())
    
    if data_dir is None:
        trait_dir = None
    else:
        trait_dir = osp.join(data_dir, trait_id)
        makedir(trait_dir)        

    pgs = []
    
    for pgs_id in tqdm(trait_info["associated_pgs_ids"], desc="Downloading PGS"):
        if trait_dir is None:
            pgs.append(download_pgs_from_catalog(pgs_id, genome_build=genome_build))
        else:
            download_pgs_from_catalog(pgs_id, trait_dir, genome_build=genome_build)
    
    return pgs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download PGS files from the PGS Catalog')

    parser.add_argument('--trait-id', dest='trait_id', type=str)
    parser.add_argument('--pgs-id', dest='pgs_id', type=str)
    parser.add_argument('--output-dir', dest='output_dir', type=str, required=True)
    parser.add_argument('--genome-build', dest='genome_build', type=str, default='GRCh37')

    args = parser.parse_args()

    assert (args.trait_id is not None) or (args.pgs_id is not None)

    if args.trait_id is not None:
        download_trait_pgs_from_catalog(args.trait_id, args.output_dir, genome_build=args.genome_build)
    elif args.pgs_id is not None:
        download_pgs_from_catalog(args.pgs_id, args.output_dir, genome_build=args.genome_build)
    else:
        raise Exception("User must provide either Trait ID or PGS ID from the PGS catalog.")
