"""
"""
from pathlib import Path
import json
import os

import numpy as np 
from sklearn import metrics

from tqdm import tqdm, trange

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from model import TemporalInfoGraph
from model.loss import jensen_shannon_mi, bce_loss
from data.tig_data_set import TIGDataset

from datetime import datetime



def train_encoder(config, name=""):
    """
    """

    #### Tracking Location ####
    date = datetime.now().strftime("%d%m%Y_%H%M")
    path = f"./output/{date}_{name}/"
    Path(path).mkdir(parents=True, exist_ok=True)
    print(path)

    #### Data Set Up ####
    data = TIGDataset(**config["data"])
    loader = DataLoader(data, **config["loader"])

    #### Model #####
    model = TemporalInfoGraph(**config["encoder"], A=data.A)

    if "print_summary" in config and config["print_summary"]:
        summary(model.to("cpu"), input_size=(2, data.A.shape[0], 300),
                batch_size=config["loader"]["batch_size"])

    model = model.to("cuda")

    loss_fn = jensen_shannon_mi
    if "loss" in config and config["loss"] == "bce":
        loss_fn = bce_loss

    train_cfg = config["encoder_training"]
    try:
        optimizer = getattr(optim, train_cfg["optimizer_name"])(
                    model.parameters(),
                    **train_cfg["optimizer"])
    except: #pylint: disable=bare-exception,bad-option-value
        optimizer = optim.Adam(model.parameters())

    #### Train Step ####
    def train_step(batch_x, batch_y):
        batch_x = batch_x.type("torch.FloatTensor").permute(0, 3, 2, 1)

        optimizer.zero_grad()

        # Returns tuple: global, local
        yhat = model(batch_x)
        loss = loss_fn(*yhat)


        if loss.isnan():
            # Loss can be nan when the batch only contains a single sample!
            return

        loss.backward()
        optimizer.step()

        return loss

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
        for batch, (batch_x, batch_y) in enumerate(tqdm(loader, leave=False,
                                                        disable=False, desc=f'Trai. Batch (Epoch: {epoch})')):

            loss = train_step(batch_x, batch_y)
            metric = tig_metric(loss_fn)

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

        np.save(path + "TIG_train_losses.npy",  epoch_loss)
        np.savez(path + "TIG_train.metrics",
                accuracy=epoch_metric["accuracy"],
                precision=epoch_metric["precision"],
                auc=epoch_metric["auc"])

        if "emb_tracking" in config and isinstance(config["emb_tracking"], int):
            if epoch % config["emb_tracking"] == 0:
                encode(loader, path, model)

        epoch_pbar.set_description(f'Epochs ({model.__class__.__name__})' +
                                   f'(Mean Acc.: {"%.2f"%np.mean(epoch_metric["accuracy"])}, ' +
                                   f'Mean Prec: {"%.2f"%np.mean(epoch_metric["precision"])}, ' +
                                   f'Mean AUC: {"%.2f"%np.mean(epoch_metric["auc"])})')

    torch.save(model, path + "TIG_.pt")

    #### Create Encodings ####
    encode(loader, path, model)


def tig_metric(loss_fn):
    yhat_norm = torch.sigmoid(loss_fn.discr_matr).detach().numpy()
    yhat_norm[yhat_norm > 0.5] = 1
    yhat_norm[yhat_norm <= 0.5] = 0

    evaluate(yhat_norm, loss_fn.mask.detach().numpy(), mode="accuracy")

    acc = evaluate(yhat_norm, loss_fn.mask.detach().numpy(), mode="accuracy")
    prec = evaluate(yhat_norm, loss_fn.mask.detach().numpy(), mode="precision")
    auc = evaluate(yhat_norm, loss_fn.mask.detach().numpy(), mode="auc")

    return {"accuracy": acc, "precision": prec, "auc": auc}


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


def encode(loader, path, model):
    emb_x = np.array([])
    emb_y = np.array([])
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Encode Data."):
            batch_x = batch_x.type("torch.FloatTensor").permute(0, 3, 2, 1)
            pred, _ = model(batch_x)
            if emb_x.size:
                emb_x = np.concatenate([emb_x, pred.detach().cpu().numpy()])
            else:
                emb_x = pred.detach().cpu().numpy()

            if emb_y.size:
                emb_y = np.concatenate([emb_y, batch_y.numpy()])
            else:
                emb_y = batch_y.numpy()

        np.savez(path+"embeddings_full", x=emb_x, y=emb_y)