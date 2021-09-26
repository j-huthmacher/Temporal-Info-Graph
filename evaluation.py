""" This file is used for the evaluation of models and baselines.
"""
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, accuracy_score
from sklearn import preprocessing

from tqdm.auto import tqdm, trange

from baseline import get_model
from utils.tig_data_set import TIGDataset
from processor import stgcn_metric


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
        data_cfg = {**config, **{"path": "./data/"}}

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

        Args:
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
            random_state = (i if portion_state is None else portion_state)
            _, x, _, y = train_test_split(x_in, y_in, stratify=y_in,
                                          test_size=portion,
                                          random_state=random_state)
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

        top1_mean, top5_mean = np.mean(top1) * 100, np.mean(top5) * 100
        pbar.set_description(f"Top1: {'{:.4f}'.format(top1_mean)} | Top2: {'{:.4f}'.format(top5_mean)}")

    if ret_model:
        return top1, top5, model
    else:
        if baseline == "mlp":
            return top1, top5, sum([len(x) for x in model.coefs_])
        else:
            return top1, top5, -1


def get_metric_stats(experiments: [tuple], exp_repetitions: int = 5):
    """ Function to get the mean and standard deviations of the metrics of an experiment along the repetitions.

        Args:
            experiments: [tuple]
                List of tuples of the form (EXPERIMENT_NAME, EXPERIMENT_PATH) with
                type (str, str).
            exp_repetitions: int
                Number of repetitions of the experiment, i.e. how of the experiment
                was repeated. The stats of the the metrics are calculated along the
                repetitions.
        Return:
            np.ndarray: List of the mean and standard deviation over the repetitions of the metrics
                        for each experiment. The list has the dimension/length of (num_experiments)
                        and each entry has the form of
                        (EXPERIMENT_NAME, [AUC_MEAN], [AUC_STD], [PREC_MEAN], [PREC_STD]) of type
                        (str, [float], [float], [float], [float]).
    """
    auc = []
    prec = []

    result = []

    for exp in experiments:
        try:
            for i in range(exp_repetitions):
                metrics = np.load(exp[1] + str(i) + "/TIG_train.metrics.npz")
                auc.append(np.squeeze(metrics["auc"]))
                prec.append(np.squeeze(metrics["precision"]))

            # E.g. "auc" has the form (num_repetitions, num_epochs). This holds also for "prec".
            result.append([exp[0], np.mean(auc, axis=0), np.std(auc, axis=0),
                           np.mean(prec, axis=0), np.std(prec, axis=0)])
            auc, prec = [], []
        # pylint: disable=broad-except
        except Exception:
            # For the case that there are no metrics stored.
            pass

    return np.array(result)


def evaluate_pytorch_model(experiments: [tuple], device: str = "cuda", repeat: int = 10,
                           verbose: bool = True, portion: int = 1):
    """ Evaluate a single pytorch model (not the TIG model). 
    
        This method is not used.

        Important: This method does not evaluate the TIG model since the TIG model is evaluated
        by downstream models. One can use this methode for example for the STGCN model, which 
        has the classifier included.

        Args:
            experiments: [tuple]
                A list of tuples, where a tuple consists of the name of the experiment and 
                the path.
            device: str
                Pytorch device, e.g. "cuda" or "cpu".
            repeat: int
                Number of repetitions of the evaluation.
            verbose: bool
                Flag to determine if the progress is printed to the console/output.
            portion: int
                Percentage defining which fraction of the data is used.
        Return:
            pd.DataFrame: Dataframe containing the results of the evaluation per experiment.
    """
    results = []

    for name, path in tqdm(experiments):
        with open(path + "config.json") as file:
            config = json.load(file)

        model = torch.load(path + "STGCN.pt").to(device)

        # Those steps aligne to the portion of data used in the training.
        data_cfg = {**config["data"], **{"name": "stgcn_50_classes", "path": "../content/"}}
        data = TIGDataset(**data_cfg)

        if "stratify" in config:
            train = data.stratify(**config["stratify"])  # num=2, num_samples=100, mode=1)
            x = np.array(train)[:, 0]
            y = np.array(train)[:, 1].astype(int)
        else:
            x = data.x
            y = data.y

        thr = int(x.shape[0] * portion)
        if portion < 1:
            classes = np.unique(y)
            num_samples = thr // len(classes)
            idx = np.array([], dtype=int)

            for cls in classes:
                idx = np.append(idx, np.where(y == cls)[0][:num_samples])

            x = x[idx]
            y = y[idx]

        # num_classes = len(np.unique(y))

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        top1 = []
        top5 = []

        pbar = trange(repeat, desc="Top1: - | Top2: - ", disable=(not verbose))

        for i in pbar:
            _, X_test, _, y_test = train_test_split(x, y, random_state=i + 50)
            val_loader = DataLoader(list(zip(X_test, y_test)), **config["loader"])

            with torch.no_grad():
                top1_batch = []
                top5_batch = []
                for batch_x, batch_y in val_loader:
                    # Input (batch_size, time, nodes, features)
                    batch_x = batch_x.type("torch.FloatTensor").to(device)
                    N, T, V, C = batch_x.size()
                    batch_x = batch_x.permute(0, 3, 1, 2).view(N, C, T, V // 2, 2)

                    # batch_x = batch_x.type("torch.FloatTensor").permute(0, 3, 2, 1).to(device)

                    yhat = model(batch_x.to(device))

                    metric = stgcn_metric(yhat, batch_y)

                    t1 = metric["top-1"]
                    t5 = metric["top-5"]
                    top1_batch.append(t1)
                    top5_batch.append(t5)

                top1.append(np.mean(top1_batch))
                top5.append(np.mean(top5_batch))

                pbar.set_description((f"Top1: {'{:.4f}'.format(np.mean(top1)*100)} "
                                      f"| Top2: {'{:.4f}'.format(np.mean(top5)*100)}"))

        results.append({
            "Model": name,
            "Top-1 Accuracy": np.mean(top1) * 100,
            "Top-1 Std.": np.std(top1) * 100,
            "Top-5 Accuracy": np.mean(top5) * 100,
            "Top-5 Std.": np.std(top5) * 100,
            "# Parameter": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "Portion": portion,
            "Num Samples": x.shape[0],
            "Classes": np.unique(y)
        })

    return pd.DataFrame(results)
