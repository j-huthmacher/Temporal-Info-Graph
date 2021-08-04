""" This file is used for the evaluation of models and baselines.
"""
import json
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, accuracy_score

from tqdm.auto import tqdm, trange

from baseline import get_model
from data.tig_data_set import TIGDataset


def load_eval_data(config: dict, path: str = None, num_classes: int = 50,
                   num_samples: int = 994):
    """ This method loads the data for evaluation based on a given data set config.

        Args:
            config: dict
                Configuration of the experiment for which the data should be loaded.
            path: str (optional)
                Path from which local data should be loaded.
            num_classe: int
                Number of classes that should be loaded.
            num_samples: int
                Determines the number of samples per class. The data set will be
                aligned to this number.

        Returns:
            (np.Array, np.Array): (x, y) where x are the features and y are the
            corresponding labels.
    """
    if path is not None:
        # Load embeddings
        x = np.load(path + "embeddings_full.npz")["x"]
        y = np.load(path + "embeddings_full.npz")["y"]
    else:
        data_cfg = {**config, **{"path": "../content/"}}

        data = TIGDataset(**data_cfg)

        if num_classes is not None:
            # 994 samples used for the kinetics skeleton dataset
            train = data.stratify(num=num_classes, num_samples=num_samples, mode=1)
            # 632 samples are used for the NTU-RGB+D dataset
            # train = data.stratify(num=num_classes, num_samples=632, mode=1)
            x = np.array(list(np.array(train)[:, 0]))
            y = np.array(train)[:, 1].astype(int)
        else:
            x = data.x
            y = data.y

        # x = np.mean(x, axis=1).reshape(x.shape[0], -1)

    return x, y


def evaluate_experiments(experiments: [tuple], baseline="mlp", repeat_baseline=5,
                         repeat_model=5, portion=1., eval_mode="model+baseline",
                         num_classes=50, unify_params=False, verbose=True,
                         repetition_mode=None):
    """ Unified method to evaluate different experiments.
        The experiments are evaluated successively.

        Parameters:
            experiments: [tuple]
                List of tuples of the form of ("MODEL_NAME", "PATH_TO_MODEL").
            baseline: str
                Name of the baseline that is used for evaluation.
                Options: ["mlp", "svm", None]
            repeat_baseline: int
                Number of repetition of train and evaluate baseline.
            portion: float
                Defining which portion of the data should be used for evaluation.
            with_baseline: bool or str
                Flag to only evaluate the encoder, only the baseline or both.
                For only evaluting the encode use True and for evaluating only
                the baseline use "only".
            num_classes: str
                Number of classes in the data. That number has to match with
                the number of classes that was used to train the TIG encoder.
            unify_params: bool
                Flag to unify the number of parameters of the baseline model
                with the number of parameters of the classifier that is appended
                to the TIG encoder. This is sometimes needed because the
                embedding size coming from the encoder can differ to the
                embedding size of the baseline data.
            verbose: bool
                Flag to print the progress.
            random_state: int
                Random state for the shuffle operation.
            repetition_mode: str
                Defines which data within one iteration is used (when the
                evaluation will be repeated). When repetition_mode is
                "same_data" we use always the same encodings and shuffle it.
                In case repetion_mode != "same_data" we use different encodings,
                i.e. the TIG model encoded multiple times (default 5) the same
                chunk of data.
    """
    results = []

    # Open the first config to get the general data configuration for the
    # first experiment. Data configuration is across all iteration from one
    # experiment the same.
    with open(f"{experiments[0][1]}/0/config.json") as file:
        config = json.load(file)

    for name, path in tqdm(experiments):
        # For each experiment load the configuration (configs differs
        # across different experiments).
        with open(f"{path}0/config.json") as file:
            config = json.load(file)

        # Defalt value for number of parameters.
        num_params = -1
        top1, top5 = [], []
        top1_baseline, top5_baseline = None, None

        ### Evaluate Model ####
        if repetition_mode == "same_data" and "model" in eval_mode:
            # Load data once and repeat evaluation on the same data.
            x, y = load_eval_data(config["data"], path=f"{path}/0/")
            top1, top5, num_params = run_evaluation(x, y, baseline,
                                                    repeat=repeat_baseline,
                                                    portion=portion,
                                                    verbose=verbose)
        elif "model" in eval_mode:
            # Load n (default n=5) times different encodings and evaluate each encoding.        
            for i in range(repeat_model):
                try:
                    x, y = load_eval_data(config["data"], path=f"{path}/{i}/")
                    t1, t5, num_params = run_evaluation(x, y, baseline,
                                                        repeat=1,
                                                        portion=portion,
                                                        portion_state=i,
                                                        verbose=verbose)
                    top1.append(t1)
                    top5.append(t5)
                # pylint: disable=broad-except
                except Exception as e:
                    # Just for debugging purpose.
                    print(e)
                    pass

        # Get percentage value of mean and standard deviation.
        top1, top1_std, top5, top5_std = (np.mean(top1) * 100,
                                          np.std(top1) * 100,
                                          np.mean(top5) * 100,
                                          np.std(top5) * 100)

        ### Evaluate Baseline ####
        if "baseline" in eval_mode:
            # Load Baseline Data
            x, y = load_eval_data(config["data"], num_classes=num_classes)
            x = np.mean(x, axis=1).reshape(x.shape[0], -1)

            if unify_params and config["model"]["architecture"][-1][-1] > 72:
                layer_size = (config["model"]["architecture"][-1][-1] - 72) + 100
                eval_temp = run_evaluation(x, y, baseline, repeat=repeat_baseline,
                                           portion=portion, hidden_layer_sizes=[layer_size])
            else:
                eval_temp = run_evaluation(x, y, baseline,
                                           repeat=repeat_baseline,
                                           portion=portion)

            top1_baseline, top5_baseline, num_params = eval_temp

        top1_baseline, top1_baseline_std = (np.mean(top1_baseline) * 100,
                                            np.std(top1_baseline) * 100)
        top5_baseline, top5_baseline_std = (np.mean(top5_baseline) * 100,
                                            np.std(top5_baseline) * 100)

        try:
            model = torch.load(f"{path}/0/TIG_.pt", map_location=torch.device('cpu'))
        except:  # noqa: E722
            model = None

        results.append({
                "Model": name + f" + {baseline}",
                "Top-1 Accuracy": top1,
                "Top-1 Std.": top1_std,
                "Top-5 Accuracy": top5,
                "Top-5 Std.": top5_std,
                "# Epochs": config["training"]["n_epochs"],
                "Architecture": config["model"]["architecture"],
                "# TIG Parameter": sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0,
                "Embedding Dimension": config["model"]["architecture"][-1][-1],
                "# Downstream Parameter": num_params,
                "Objective": config["loss"],
                "Model Cfg": config["model"],
                "Portion": portion,
                "Batch Size": config["loader"]["batch_size"],
                "Num Samples": int(x.shape[0] * portion),
                "Samples Per Class": np.unique(y, return_counts=True)[1][0],
                "Classes": np.unique(y).shape[0],
                "Timestamp": datetime.now()
            })

        results.append({
                "Model": baseline,
                "Top-1 Accuracy": top1_baseline,
                "Top-1 Std.": top1_baseline_std,
                "Top-1 Relative": "-",
                "Top-5 Accuracy": top5_baseline,
                "Top-5 Std.": top5_baseline_std,
                "Top-5 Relative": "-",
                "# Epochs": "-",
                "Architecture": "-",
                "# TIG Parameter": "-",
                "Embedding Dimension": x.shape[1],
                "# Downstream Parameter": num_params,
                "Objective": "-",
                "Model Cfg": "-",
                "Portion": portion,
                "Batch Size": "-",
                "Num Samples": int(x.shape[0] * portion),
                "Samples Per Class": np.unique(y, return_counts=True)[1][0],
                "Classes": np.unique(y).shape[0],
                "Timestamp": datetime.now()
            })

        results = pd.DataFrame(results)

        if top1_baseline is not None:
            results["Top-1 Relative"] = results["Top-1 Accuracy"] - top1_baseline
        if top5_baseline is not None:
            results["Top-5 Relative"] = results["Top-5 Accuracy"] - top5_baseline
    
    columns_ordered = ["Model", "Top-1 Accuracy", "Top-1 Std.", "Top-1 Relative", "Top-5 Accuracy", "Top-5 Std.", "Top-5 Relative",
                       "# Epochs", "Architecture", "# TIG Parameter", "Embedding Dimension", "# Downstream Parameter", "Objective",
                       "Model Cfg", "Portion", "Batch Size", "Num Samples", "Samples Per Class", "Classes", "Timestamp"]

    return results[columns_ordered]


def run_evaluation(x_in: np.ndarray, y_in: np.ndarray, baseline: str = "svm",
                   repeat: int = 10, ret_model: bool = False, verbose: bool = True,
                   portion: float = 1., portion_state: int = None, **baseline_params):
    """ Executes a concrete evaluation run.

        Args:
            x_in: np.ndarray
                Numpy array with the input data on which the evaluation should
                be executed.
            y_in: np.ndarray
                Corresponding labels to the input data in x_in as numpy array.
            baseline: str
                Determines which baseline is used. Options ["svm", "mlp"]
            repeat: int
                Number of repetitions that will be performed.
            ret_model: bool
                Flag to decide if the model will be returned.
            verbose: bool
                Flag to decide if the process should be printed to the console.
            portion: float
                Determines which fraction of the data is used. E.g. 1 for the
                whole data and 0.1 for 10%.
            portion_state: int
                Random state when splitting the date into the portion. If None
                the number of iteration is used for reprodiciblity.
            **baseline_params:
                Parameters for the sklean baseline.

        Returns:
            (top1-accuracy, top5-accuracy, -1) or
            (top1-accuracy, top5-accuracy, num_model_coeff)
    """
    top1 = []
    top5 = []

    pbar = trange(repeat, desc="Top1: - | Top2: - ", disable=(not verbose))

    for i in pbar:
        if portion < 1:
            # x, y = x[idx], y[idx]
            # x, y = get_portion(x, y, portion)
            _, x, _, y = train_test_split(x_in, y_in, stratify=y_in,
                                          test_size=portion,
                                          random_state=i if portion_state is None else portion_state)
        else:
            x, y = x_in, y_in

        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=i)
        # pylint: disable=broad-except
        except Exception:
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=i)

        model = get_model(baseline, verbose=0, random_state=i, **baseline_params)
        model.fit(x_train, y_train)

        if baseline == "svm":
            yhat = model.decision_function(x_test)
        elif baseline == "mlp":
            yhat = model.predict_proba(x_test)

        try:
            top1.append(top_k_accuracy_score(y_test, yhat, k=1))
            top5.append(top_k_accuracy_score(y_test, yhat, k=5))
        # pylint: disable=broad-except
        except Exception:
            top1.append(accuracy_score(y_test, model.predict(x_test)))
            top5.append(-1)

        top1, top5 = np.mean(top1)*100, np.mean(top5)*100
        pbar.set_description(f"Top1: {'{:.4f}'.format(top1)} | Top2: {'{:.4f}'.format(top5)}")

    if ret_model:
        return top1, top5, model
    else:
        if baseline == "mlp":
            return top1, top5, sum([len(x) for x in model.coefs_])
        else:
            return top1, top5, -1
