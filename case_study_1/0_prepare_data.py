import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__)),
                         "data_preparation/4_generate_datasets/"))
from create_datasets import create_prs_dataset
from magenpy.utils.system_utils import makedir

phenotypes = ['TST', 'URT', 'CRTN']
biobanks = ['ukbb', 'cartagene']
prop_test = 0.3  # Proportion of test samples

for bbk in biobanks:
    for pheno in phenotypes:

        try:
            prs_dataset = create_prs_dataset(
                bbk,
                pheno,
                "1kghdp",
                "1kghdp",
                #ancestry_subset='EUR',  # Restrict to European samples
            )
        except Exception as e:
            print(e)
            continue

        output_dir = f"data/case_1/harmonized_data/{pheno}/{bbk}/"

        makedir(output_dir)

        # Save the entire dataset:
        prs_dataset.save(osp.join(output_dir, "full_data.pkl"))

        # Split the dataset into training and testing sets:
        train_data, test_data = prs_dataset.train_test_split(test_size=prop_test)

        train_data.save(osp.join(output_dir, "train_data.pkl"))
        test_data.save(osp.join(output_dir, "test_data.pkl"))

