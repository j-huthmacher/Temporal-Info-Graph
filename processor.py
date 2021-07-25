""" The processor script takes care of the general training procedure of different models.

In this file you will find among others:
* batch iteration
* train steps / optimization steps
* tracking of mertrics such as accuracy, auc, or precision


"""
from pathlib import Path
import json
import os
import logging

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, accuracy_score
from sklearn import preprocessing

from tqdm.auto import tqdm, trange

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from model.tig import TemporalInfoGraph
from model.st_gcn_aaai18.stgcn_model import ST_GCN_18
from model.loss import jensen_shannon_mi, bce_loss
from data.tig_data_set import TIGDataset

from baseline import get_model

from datetime import datetime

def train_baseline(x, y, baseline = "svm"):
    
    x_train,x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    model = get_model(baseline, verbose=0)
    model.fit(x_train, y_train)

    if baseline == "svm":
        yhat = model.decision_function(x_test)
    elif baseline == "mlp":
        yhat = model.predict_proba(x_test)
    
    try:
        return top_k_accuracy_score(y_test, yhat, k = 1), top_k_accuracy_score(y_test, yhat, k = 5)
    except:
        return accuracy_score(y_test, model.predict(x_test)), -1


def run_baseline(data_cfg, path, baseline="svm", num_iter=10):
    #### Tracking Location ####
    log = logging.getLogger('TIG_Logger')
    log.info(f"Output path: " + path)
    
    data = TIGDataset(**data_cfg)  
    
    x = np.array(data.x).reshape(np.array(data.x).shape[0], -1)
    y = np.array(data.y).astype(int)
    
    top5 = []
    top1 = []
    log.info("Start baseline training")    
    
    pbar = trange(num_iter, desc="Top1: - | Top2: - ")
    
    for i in pbar:
        t1, t5 = train_baseline(x, y, baseline)
        top1.append(t1)
        top5.append(t5)
        pbar.set_description(f"Top1: {np.mean(top1)*100} | Top2: {np.mean(top5)*100} ")
    
    log.info("Save baseline accuracies...")
    np.save(path + f"top1_{baseline}.npy", top1)
    np.save(path + f"top5_{baseline}.npy", top5)
    
    return np.mean(top1)*100, np.std(top1)*100, np.mean(top5)*100, np.std(top5)*100
    

def train_stgcn(config, path):
    """
    """

    #### Tracking Location ####
    log = logging.getLogger('TIG_Logger')
    log.info(f"Output path: " + path)

    #### Data Set Up ####
    data = TIGDataset(**config["data"])

    if "stratify" in config:
        train = data.stratify(**config["stratify"])#num=2, num_samples=100, mode=1)
        x = np.array(train)[:, 0]
        y = np.array(train)[:, 1].astype(int)
    else:
        x = data.x
        y = data.y

    num_classes = len(np.unique(y))

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

    train_loader = DataLoader(list(zip(X_train, y_train)), **config["loader"])
    val_loader = DataLoader(list(zip(X_test, y_test)), **config["loader"])

    #### Model #####
    graph_cfg = {
        "layout": 'openpose',
        "strategy": 'spatial'
    }
    model_cfg = config["model"] if "model" in config else {}

    model = TemporalInfoGraph(**model_cfg, mode="classify", num_classes=num_classes, A=data.A[:18, :18])
    # model = ST_GCN_18(2, num_classes, graph_cfg, edge_importance_weighting=True)
    # Data set has only 2 classes but the classes 12 and 49 therfore we need a larger 1 hot vector

    # if "print_summary" in config and config["print_summary"]:
    #     summary(model.to("cuda"), input_size=(2, data.A.shape[0], 300),
    #             batch_size=config["loader"]["batch_size"])

    model = model.to("cuda")

    log.info("Model parameter: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    loss_fn = nn.CrossEntropyLoss()

    train_cfg = config["training"]
    try:
        optimizer = getattr(optim, train_cfg["optimizer_name"])(
                    model.parameters(),
                    **train_cfg["optimizer"])
    except: #pylint: disable=bare-exception,bad-option-value
        optimizer = optim.Adam(model.parameters())

    #### Train Step ####
    def train_step(batch_x, batch_y):

        # Input (batch_size, time, nodes, features)
        batch_x = batch_x.type("torch.FloatTensor").to("cuda")
        N, T, V, C = batch_x.size()
        batch_x = batch_x.permute(0, 3, 1, 2).view(N, C, T, V//2, 2)

        # batch_x = batch_x.type("torch.FloatTensor").permute(0, 3, 2, 1).to("cuda")

        optimizer.zero_grad()

        # Returns tuple: global, local
        yhat = model(batch_x).to("cuda")
        loss = loss_fn(yhat, batch_y.type("torch.LongTensor").to("cuda"))

        loss.backward()
        optimizer.step()

        return yhat, loss

    with open(f'{path}/config.json', 'w') as fp:
        json.dump({
            **config,
            **{
                "train_data_size": len(train_loader.dataset),
                "val_data_size": len(val_loader.dataset),
                "exec_dir": os.getcwd(),
                "optimzer": str(optimizer),
            }}, fp)

    #### Train ####
    epoch_loss = []
    epoch_metric = {}
    epoch_val_loss = []
    epoch_val_metric = {}
    epoch_pbar = trange(train_cfg["n_epochs"], disable=(not train_cfg["verbose"]),
                        desc=f'Epochs ({model.__class__.__name__})')
    for epoch in epoch_pbar:
        model.train()

        batch_loss = []
        batch_metric = {}
        for batch, (batch_x, batch_y) in enumerate(tqdm(train_loader, leave=False,
                                                        disable=False, desc=f'Trai. Batch (Epoch: {epoch})')):

            yhat, loss = train_step(batch_x, batch_y)
            metric = stgcn_metric(yhat, batch_y)

            batch_loss.append(torch.squeeze(loss).item())
            for key in metric:
                batch_metric[key] = (batch_metric[key] + [metric[key]]
                                     if "key" in batch_metric
                                     else [metric[key]])


        epoch_loss.append(np.mean(batch_loss))
        for key in batch_metric:
            metric_val = np.mean(batch_metric[key])
            epoch_metric[key] = (epoch_metric[key] + [metric_val]
                                 if key in epoch_metric
                                  else [metric_val])

        np.save(path + "STGCN_train_losses.npy",  epoch_loss)
        np.savez(path + "STGCN_train.metrics",
                 top1=epoch_metric["top-1"],
                 top5=epoch_metric["top-5"])

        ########################################################################

        batch_loss = []
        batch_metric = {}
        with torch.no_grad():
            for batch, (batch_x, batch_y) in enumerate(tqdm(val_loader, leave=False,
                                                            disable=False, desc=f'Val. Batch (Epoch: {epoch})')):
                # Input (batch_size, time, nodes, features)
                batch_x = batch_x.type("torch.FloatTensor").to("cuda")
                N, T, V, C = batch_x.size()
                batch_x = batch_x.permute(0, 3, 1, 2).view(N, C, T, V//2, 2)

                # batch_x = batch_x.type("torch.FloatTensor").permute(0, 3, 2, 1).to("cuda")

                yhat = model(batch_x.to("cuda"))
                loss = loss_fn(yhat, batch_y.type("torch.LongTensor").to("cuda"))
                metric = stgcn_metric(yhat, batch_y)

                batch_loss.append(torch.squeeze(loss).item())
                for key in metric:
                    batch_metric[key] = (batch_metric[key] + [metric[key]]
                                        if "key" in batch_metric
                                        else [metric[key]])


        epoch_val_loss.append(np.mean(batch_loss))
        for key in batch_metric:
            metric_val = np.mean(batch_metric[key])
            epoch_val_metric[key] = (epoch_val_metric[key] + [metric_val]
                                     if key in epoch_val_metric
                                     else [metric_val])

        np.save(path + "STGCN_val_losses.npy",  epoch_val_loss)
        np.savez(path + "STGCN_val.metrics",
                 top1=epoch_val_metric["top-1"],
                 top5=epoch_val_metric["top-5"])

        ########################################################################

        epoch_pbar.set_description(f'Epochs ({model.__class__.__name__}) ' +
                                   f'Mean Top1: {"%.2f"%np.mean(epoch_val_metric["top-1"])}, ' +
                                   f'Mean Top5: {"%.2f"%np.mean(epoch_val_metric["top-5"])}, ' )

    torch.save(model, path + "STGCN.pt")


def train_tig(config: dict, path: str):
    """ Is the training function for the TIG model and it includes the tracking of the metrics.

        Parameters:
            config: dict
                Dictionary of configuration paramters
            path: str
                Path where all artifacts that are created during training are stored.

        Return:
            None
    """
    # Instantiate logger
    log = logging.getLogger('TIG_Logger')
    log.info("Output path: %s", path)

    #### Set Up Data ####
    data = TIGDataset(**config["data"])

    if "stratify" in config:
        train = data.stratify(**config["stratify"])
        loader = DataLoader(train, **config["loader"])
    else:
        loader = DataLoader(data, **config["loader"])

    #### Model #####
    model_cfg = config["model"] if "model" in config else {}
    model = TemporalInfoGraph(**model_cfg, A=data.A)

    model = model.to("cuda")

    log.info("Model parameter: %s",
             str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Default loss function is JSD MI estimate
    loss_fn = jensen_shannon_mi
    if "loss" in config and config["loss"] == "bce":
        # Alternatively one can use the binary cross-entropy loss
        loss_fn = bce_loss

    train_cfg = config["training"]
    try:
        # Get optimizer class from PyTorch dynamically using the name 
        optimizer = getattr(optim, train_cfg["optimizer_name"])
        # Instantiate PyTorch optimizer
        optimizer = optimizer(model.parameters(), **train_cfg["optimizer"])
    # pylint: disable=bare-except,bad-option-value
    except:  # noqa: E722
        optimizer = optim.Adam(model.parameters())

    #### Train Step ####
    def train_step(batch_x: torch.Tensor):
        """ Definition of a single train step.

            Parameters:
                batch_x: torch.Tensor
                    Batch of feature tensors
            Return:
                torch.Tensor: Tensor of size (N, 1) containing the (batch) loss.
        """
        batch_x = batch_x.type("torch.FloatTensor").to("cuda")
        # N: num. samples; T: num. timesteps; V: num. persons; C: num. features
        N, T, V, C = batch_x.size()
        # Each frame always contain two skeletons. Here we split it into two separate items.
        batch_x = batch_x.permute(0, 3, 1, 2).view(N, C, T, V // 2, 2)

        optimizer.zero_grad()

        # Returns tuple: global emb., local emb.
        yhat = model(batch_x)
        loss = loss_fn(*yhat)

        if loss.isnan():
            # Loss can be nan when the batch only contains a single sample!
            # Just ignore this case.
            return None

        loss.backward()
        optimizer.step()

        return loss

    # Before starting the training, extend the config among others by the optimizer
    with open(f'{path}/config.json', 'w') as fp:
        json.dump({
            **config,
            **{
                "train_data_size": len(loader.dataset),
                "exec_dir": os.getcwd(),
                "optimzer": str(optimizer),
            }}, fp)

    #### Train ####
    epoch_loss = []
    epoch_metric = {}
    epoch_pbar = trange(train_cfg["n_epochs"], disable=(not train_cfg["verbose"]),
                        desc=f'Epochs ({model.__class__.__name__})')
    for epoch in epoch_pbar:
        model.train()

        batch_loss = []
        batch_metric = {}
        # The loader returns a tuple of (batch_x, batch_y)
        batch_bar = tqdm(loader, leave=False, disable=False,
                         desc=f'Trai. Batch (Epoch: {epoch})')
        for _, (batch_x, _) in enumerate(batch_bar):

            loss = train_step(batch_x)
            batch_loss.append(torch.squeeze(loss).item())

            # Calculate metrics for current batch
            metric = tig_metric(loss_fn)            
            for key in metric:
                # Stack metrics per batch to get an array for each metric
                batch_metric[key] = (batch_metric[key] + [metric[key]]
                                     if key in batch_metric
                                     else [metric[key]])

        epoch_loss.append(np.mean(batch_loss))

        # Calculate metric on epoch level by averaging the metrics from each single step
        for key in batch_metric:
            metric_val = np.mean(batch_metric[key])
            epoch_metric[key] = (epoch_metric[key] + [metric_val]
                                 if key in epoch_metric
                                 else [metric_val])

        np.save(path + "TIG_train_losses.npy", epoch_loss)
        np.savez(path + "TIG_train.metrics",
                 accuracy=epoch_metric["accuracy"],
                 precision=epoch_metric["precision"],
                 auc=epoch_metric["auc"])

        # Optionally one can save the embeddings after each epoch to have
        # intermediate results in case of an exceptional error.
        if "emb_tracking" in config and isinstance(config["emb_tracking"], int):
            if epoch % config["emb_tracking"] == 0:
                encode(loader, path, model, verbose=False)

        epoch_pbar.set_description((f'Epochs ({model.__class__.__name__})'
                                    f'(Mean Acc.: {"%.2f"%np.mean(epoch_metric["accuracy"])}, '
                                    f'Mean Prec: {"%.2f"%np.mean(epoch_metric["precision"])}, '
                                    f'Mean AUC: {"%.2f"%np.mean(epoch_metric["auc"])})'))

    torch.save(model, path + "TIG_Model.pt")

    #### Create Encodings ####
    encode(loader, path, model)

# TODO: Check if it is used, Cleand code
def tig_metric(loss_fn):
    yhat_norm = torch.sigmoid(loss_fn.discr_matr).detach().numpy()
    yhat_norm[yhat_norm > 0.5] = 1
    yhat_norm[yhat_norm <= 0.5] = 0

    evaluate(yhat_norm, loss_fn.mask.detach().numpy(), mode="accuracy")

    acc = evaluate(yhat_norm, loss_fn.mask.detach().numpy(), mode="accuracy")
    prec = evaluate(yhat_norm, loss_fn.mask.detach().numpy(), mode="precision")
    auc = evaluate(yhat_norm, loss_fn.mask.detach().numpy(), mode="auc")

    return {"accuracy": acc, "precision": prec, "auc": auc}


def stgcn_metric(yhat, batch_y):
    yhat = yhat.cpu().detach().numpy()

    # idx = np.argsort(yhat)[::-1]

    acc = evaluate(yhat, batch_y.cpu().detach().numpy(), mode="top-k")
    # prec = evaluate(yhat, batch_y.cpu().detach().numpy(), mode="precision")
    # auc = evaluate(yhat, batch_y.cpu().detach().numpy(), mode="auc")

    return {"top-1": acc[0], "top-5": acc[1]}


def evaluate(predictions: np.array, labels: np.array, mode: str = "top-k"):
    """ Function to calculate the evaluation metric.
            Parameters:
                predictions: np.array
                    Array with the predicted values.
                labels: np.array
                    Corresponding ground truth for the predicted values.
                mode: str (not used yet)
                    Mode to decide with evaluation should be used.
            Return:
                tuple: tuple containing the evaluation metrics.
        """
    if mode == "top-k":
        # Sort predictions
        predictions = np.argsort(predictions, axis=1)[:, ::-1]

        # Here predictions need to be sorted!
        k = 1  # Top-k
        correct = np.sum([l in pred for l, pred in zip(
            labels, np.asarray(predictions)[:, :k])])
        top1 = (correct/len(labels))

        k = 5  # Top-k
        correct = np.sum([l in pred for l, pred in zip(
            labels, np.asarray(predictions)[:, :k])])
        top5 = (correct/len(labels))

        metric = (top1, top5)
    elif mode == "accuracy":
        metric = (predictions.flatten() == labels.flatten()
                  ).sum() / len(labels.flatten())
    elif mode == "precision":
        # Works only for
        TP = ((predictions.flatten() == 1) & (labels.flatten() == 1)).sum()
        FP = ((predictions.flatten() == 1) & (labels.flatten() == 0)).sum()
        metric = TP / (TP+FP) if TP != 0 else 0
    elif mode == "recall":
        # Works only for
        TP = ((predictions == 1) & (labels == 1)).sum()
        FN = ((predictions == 0) & (labels == 0)).sum()
        metric = TP / (TP+FN) if TP != 0 else 0
    elif mode == "auc":
        fpr, tpr, thresholds = metrics.roc_curve(
            labels.flatten(), predictions.flatten())
        metric = metrics.auc(fpr, tpr)

    # accuracy
    return metric


def encode(loader, path, model, verbose=True):
    emb_x = []
    emb_y = []
    with torch.no_grad():
        model.eval()
        for i, (batch_x, batch_y) in enumerate(tqdm(loader, desc="Encode Data.")):
            batch_x = batch_x.type("torch.FloatTensor").to("cuda")
            N, T, V, C = batch_x.size()
            batch_x = batch_x.permute(0, 3, 1, 2).view(N, C, T, V // 2, 2)

            # NTU RGB + D
            # N, T, V, C = batch_x.size()
            # batch_x = batch_x.permute(0, 3, 1, 2).view(N, C, T, V, 1)

            # batch_x = batch_x.type("torch.FloatTensor").permute(0, 3, 2, 1)
            pred, _ = model(batch_x)

            emb_x.append(pred.detach().cpu().numpy())        
            emb_y.append(batch_y.numpy())

        emb_x = np.concatenate(emb_x)
        emb_y = np.concatenate(emb_y)

        np.savez(path + "embeddings_full", x=emb_x, y=emb_y)
    
    return emb_x, emb_y