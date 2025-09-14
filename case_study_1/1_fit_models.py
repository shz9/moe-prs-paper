import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__)),
                         "model/"))
from train_models import train_all_models
from PRSDataset import PRSDataset
from magenpy.utils.system_utils import makedir

phenotypes = ['TST', 'URT', 'CRTN']
biobanks = ['ukbb', 'cartagene']

for pheno in phenotypes:
    for bbk in biobanks:

        dataset_path = f"data/case_1/harmonized_data/{pheno}/{bbk}/train_data.pkl"

        try:
            prs_dataset = PRSDataset.from_pickle(dataset_path)
        except Exception as e:
            print(e)
            continue

        trained_models = train_all_models(prs_dataset, {}, {})

        output_dir = osp.dirname(dataset_path).replace('harmonized_data', 'trained_models')
        dataset_name = osp.basename(dataset_path).replace('.pkl', '')

        output_dir = osp.join(output_dir, dataset_name)

        makedir(output_dir)

        print("> Saving trained models to:\n\t", output_dir)

        for model_name, model in trained_models.items():
            model.save(osp.join(output_dir, f'{model_name}.pkl'))

