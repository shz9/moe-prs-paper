import sys
import os.path as osp
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(parent_dir, "model/"))
import pickle
import glob
import numpy as np
import pandas as pd
import argparse



def predict_dataset(prs_dataset, models, backend='numpy'):

    if isinstance(prs_dataset, str):
        prs_dataset = load_dataset(prs_dataset, backend=backend)
    else:
        prs_dataset.set_backend(backend)

    preds = {}

    for model_name, model in models.items():

        try:
            preds[model_name] = model.predict(prs_dataset.get_covariates(),
                                              prs_dataset.get_expert_predictions())
        except ValueError as e:
            preds[model_name] = model.predict(prs_dataset.get_covariates(),
                                              prs_dataset.get_expert_predictions(),
                                              prs_dataset.get_covariates())
        except TypeError as e:
            print(model_name)
            print(e)
            prs_dataset.set_backend('torch')
            preds[model_name] = model.predict(prs_dataset.get_covariates(),
                                              prs_dataset.get_expert_predictions()).detach().numpy()
            prs_dataset.set_backend('numpy')

    return preds


def predict_proba_dataset(prs_dataset, models, backend='numpy'):

    if isinstance(prs_dataset, str):
        prs_dataset = load_dataset(prs_dataset, backend=backend)
    else:
        prs_dataset.set_backend(backend)

    proba = {}

    for model_name, model in models.items():

        try:
            proba[model_name] = model.predict_proba(prs_dataset.get_covariates(),
                                                    prs_dataset.get_expert_predictions())
        except TypeError:
            prs_dataset.set_backend('torch')
            proba[model_name] = model.predict_proba(prs_dataset.get_covariates(),
                                                    prs_dataset.get_expert_predictions()).detach().numpy()
            prs_dataset.set_backend('numpy')

    return proba


def generate_predictions(test_data):

    model_dirs = "model/saved_models/" + "/".join(test_data.split("/")[-3:])
    model_dirs = model_dirs.replace(".dat", "").replace("test_", "train_")

    numpy_models = glob.glob(f"{model_dirs}/numpy_models/*.npy")
    pytorch_models = glob.glob(f"{model_dirs}/PyTorch_models/*.ckpt")
    base_models = glob.glob(f"{model_dirs}/base_models/*.pkl")

    if len(numpy_models) + len(pytorch_models) + len(base_models) < 1:
        raise Exception("Could not find trained models in:", model_dirs)

    dataset = load_dataset(test_data)

    res = {}

    print("> Processing base models...")

    for m in base_models:

        model_name = osp.basename(m).replace(".pkl", "")

        print("Processing:", model_name)

        res[model_name] = predict_dataset(
                    dataset,
                    PRSTuningModel.from_saved_model(m),
                    backend='numpy'
                )

    print("> Processing numpy models...")

    for m in numpy_models:

        model_name = osp.basename(m).replace(".npy", "")

        print("Processing:", model_name)

        try:
            if 'MultiPRS' in model_name:
                # Need to figure it out...
                continue
            elif 'MoEssi' in model_name:
                
                model = LinearGate_MoEssi.from_saved_model(m)
                model.repeat = (model.gate_params.shape[1] + 1) // dataset.n_experts
                res[model_name] = predict_dataset(
                    dataset,
                    model,
                    backend='numpy'
                )

            else:
                res[model_name] = predict_dataset(
                    dataset,
                    LinearGate_MoE.from_saved_model(m),
                    backend='numpy'
                )
        except Exception as e:
            print(e)
            continue

    print("> Processing PyTorch models...")

    for m in pytorch_models:

        model_name = osp.basename(m).replace(".ckpt", "")
        print("Processing:", model_name)

        # Load the model:
        gate_model = load_saved_model(m,
                                      dataset.n_covariates, dataset.n_experts,
                                      'linear' in model_name)

        res[model_name] = predict_dataset(
            dataset,
            gate_model,
            backend='torch'
        )

    return pd.DataFrame(res)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predict for combined models")

    parser.add_argument("--test-data", dest="test_data", type=str, required=True)

    args = parser.parse_args()
    preds = generate_predictions(args.test_data)
    preds.to_csv(f"data/combined_predictions/{args.test_data}.csv", index=False)

