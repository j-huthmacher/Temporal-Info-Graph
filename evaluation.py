
import json

import torch
from torch.utils.data import DataLoader
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, accuracy_score
from sklearn import preprocessing

from tqdm.auto import tqdm, trange

from baseline import get_model
from experiments import Experiment
from data.tig_data_set import TIGDataset
from processor import stgcn_metric

def evaluate_pytorch_model(experiments: [tuple], device: str = "cuda", repeat=10, verbose=True, portion=1):
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

        top1_test = []
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

                pbar.set_description(f"Top1: {'{:.4f}'.format(np.mean(top1)*100)} | Top2: {'{:.4f}'.format(np.mean(top5)*100)} ")

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

def load_eval_data(config, path=None, num_classes=50):
    """
    """
    if path is not None:
        # Load embeddings
        x = np.load(path + "embeddings_full.npz")["x"]
        y = np.load(path + "embeddings_full.npz")["y"]
    else:
        # Load baseline data
        data_cfg = {**config, **{"name": "stgcn_50_classes", "path": "../content/"}}
        data = TIGDataset(**data_cfg)

        if num_classes < 50:
            train = data.stratify(num=num_classes, num_samples=994, mode=1)
            x = np.array(list(np.array(train)[:, 0]))
            y = np.array(train)[:, 1].astype(int)
        else:
            x = data.x
            y = data.y

        x = np.mean(x, axis=1).reshape(x.shape[0], -1)
    
    return x, y


def get_portion(x, y, portion):
    """
    """

    thr = int(x.shape[0] * portion)
    classes = np.unique(y)
    num_samples = thr // len(classes)
    idx = np.array([], dtype=int)

    for cls in classes:
        idx = np.append(idx, np.where(y==cls)[0][:num_samples])

    return x[idx], y[idx]
   

def evaluate_experiments(experiments: [tuple], baseline="mlp", repeat=10, portion=1,
                         num_classes=50, unify_params=True, verbose=False):
    """ Unified method evaluate different experiments
    """    

    results = []

    for name, path in tqdm(experiments):
        # exp = Experiment(path)
        with open(path + "config.json") as file:
            config = json.load(file)

        x, y = load_eval_data(config, path=path)

        # x = np.load(path + "embeddings_full.npz")["x"]
        # y = np.load(path + "embeddings_full.npz")["y"]

        if portion < 1:
            x, y = get_portion(x, y, portion)

        top1, top5, num_params = run_evaluation(x, y, baseline, repeat=repeat,
                                                verbose=verbose)

        try:
            model = torch.load(path + "TIG_.pt", map_location=torch.device('cpu'))
        except:
            model = None

        results.append({
            "Model": name + f" + {baseline}",
            "Top-1 Accuracy": np.mean(top1) * 100,
            "Top-1 Std.": np.std(top1) * 100,
            "Top-5 Accuracy": np.mean(top5) * 100,
            "Top-5 Std.": np.std(top5) * 100,
            "Embedding Dimension": config["model"]["architecture"][-1][-1],
            "# TIG Parameter": sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0,
            "# Downstream Parameter": num_params,
            "Objective": config["loss"],
            "Architecture": config["model"]["architecture"],
            "Model Cfg": config["model"],
            "Portion": portion,
            "Num Samples": x.shape[0],
            "Samples Per Class": np.unique(y, return_counts=True)[1][0],
            "Classes": np.unique(y)
        })

    # Baseline
    # data_cfg = {**config["data"], **{"name": "stgcn_50_classes", "path":"../content/"}}
    # data = TIGDataset(**data_cfg)

    # if num_classes < 50:
    #     train = data.stratify(num=num_classes, num_samples=994, mode=1)
    #     x = np.array(list(np.array(train)[:, 0]))
    #     y = np.array(train)[:, 1].astype(int)
    # else:
    #     x = data.x
    #     y = data.y

    # x = np.mean(x, axis=1).reshape(x.shape[0], -1)

    x, y = load_eval_data(config, path=path, num_classes=num_classes)

    if portion < 1:
        x, y = get_portion(x, y, portion)

    if unify_params and config["model"]["architecture"][-1][-1] > 72:
        layer_size = (config["model"]["architecture"][-1][-1] - 72) + 100
        top1, top5, num_params = run_evaluation(x, y, baseline, repeat=repeat,
                                                hidden_layer_sizes=[layer_size])
    else:
        top1, top5, num_params = run_evaluation(x, y, baseline, repeat=repeat)

    results.append({
            "Model": baseline,
            "Top-1 Accuracy": np.mean(top1) * 100,
            "Top-1 Std.": np.std(top1) * 100,
            "Top-5 Accuracy": np.mean(top5) * 100,
            "Top-5 Std.": np.std(top5) * 100,
            "Embedding Dimension": x.shape[1],
            "# TIG Parameter": "-",
            "# Downstream Parameter": num_params,
            "Objective": "-",
            "Architecture": "-",
            "Model Cfg": "-",
            "Portion": portion,
            "Num Samples": x.shape[0],
            "Samples Per Class": np.unique(y, return_counts=True)[1][0],
            "Classes": np.unique(y)
        })

    return pd.DataFrame(results)


def run_evaluation(x, y, baseline = "svm", repeat=10, ret_model=False,
                   verbose=True, **baseline_params):

    top1 = []
    top5 = []

    pbar = trange(repeat, desc="Top1: - | Top2: - ", disable=(not verbose))

    for i in pbar:
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=i+50)
        except:
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=i+50)
        
        model = get_model(baseline, verbose=0, random_state=i+50, **baseline_params)
        model.fit(x_train, y_train)

        if baseline == "svm":
            yhat = model.decision_function(x_test)
        elif baseline == "mlp":
            yhat = model.predict_proba(x_test)

        try:
            top1.append(top_k_accuracy_score(y_test, yhat, k = 1))
            top5.append(top_k_accuracy_score(y_test, yhat, k = 5))
        except Exception as e:
            top1.append(accuracy_score(y_test, model.predict(x_test)))
            top5.append(-1)

        pbar.set_description(f"Top1: {'{:.4f}'.format(np.mean(top1)*100)} | Top2: {'{:.4f}'.format(np.mean(top5)*100)} ")

    # try:
    #     print("Number of paramter: ", sum([len(x) for x in model.coefs_]))
    # except:
    #     pass

    if ret_model:
        return top1, top5, model
    else:
        return top1, top5, sum([len(x) for x in model.coefs_])
