#pylint: disable=line-too-long
""" Main file.

    @author: jhuthmacher
"""
import os
import json
import signal
import sys
from pathlib import Path

import argparse
import torch
from torch.utils.data import DataLoader
import yaml
from datetime import datetime
import numpy as np

from tqdm import trange

#pylint: disable=import-error
from tracker import Tracker
from experiments import  experiment
from config.config import log
from data.tig_data_set import TIGDataset
from experiments import Experiment
from baseline import train_baseline, get_model
from processor import train_tig, train_stgcn

from config.config import create_logger


from sklearn.metrics import accuracy_score, top_k_accuracy_score


#### Set Up CLI ####
parser = argparse.ArgumentParser(prog='tig', description='Temporal Info Graph')

#### Model/Experiment CLI ####
parser.add_argument('--config', dest='config',
                    help='Defines which configuration should be usd. Can be a name, .json or .yml file.')
parser.add_argument('--name', dest='name', default="TIG_Experiment",
                    help='Name of the experiment.')
parser.add_argument('--tracking', dest='tracking', default="remote",
                    help='[remote, local], default: remote')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Flag to select trainings mode.')
parser.add_argument('--model', dest='model', default="tig",
                    help='[tig, stgcn]')


parser.add_argument('--downstream', dest='downstream', action='store_true',
                    help='Flag to determine if the downstream training should be executed.')
parser.add_argument('--disable_local_store', dest='disable_local_store', action='store_true',
                    help='Flag to determine if the models should be locally stored. Default: Models are stored locally.')
#### Data CLI ####
parser.add_argument('--prep_data', dest='prep_data', action="store_true",
                    help='Prepare data.')
#### Baseline ####
parser.add_argument('--baseline', dest='baseline', action="store_true",
                    help='Execute baseline')
parser.add_argument('--data', dest='data', default="stgcn_50_classes",
                    help='Name of the data set that should be used.')

# parser.add_argument('--type', dest='type', action="str",
#                     help='Prepare data.')

args = parser.parse_args()

#### Load Configuration ####
name = args.name
config = {}
if args.config is not None:
    if ".yml" in args.config or ".yaml" in args.config:
        with open(args.config) as file:
            name = args.name
            config = yaml.load(file, Loader=yaml.FullLoader)
    elif ".json" in args.config:
        with open(args.config) as file:
            name = args.name
            config = json.load(file)
    else:
        with open("./experiments/config_repo.yml") as file:
            name += f"_{args.config}"
            config = yaml.load(file, Loader=yaml.FullLoader)[args.config]
else:
    with open("./experiments/config_repo.yml") as file:
        name += "_standard"
        config = yaml.load(file, Loader=yaml.FullLoader)["standard"]

#### Execution ####
if args.train:
    if "name" in config:
        name += f"_{config['name']}"

    # For reproducibility
    torch.manual_seed(0)
    config["seed"] = 0

    #### Tracking Location ####
    date = datetime.now().strftime("%d%m%Y_%H%M")
    path = f"./output/{date}_{name}/"
    Path(path).mkdir(parents=True, exist_ok=True)

    log = create_logger(path)

    if args.model == "tig":
        train_tig(config, path)
    elif args.model == "stgcn":
        for i in range(10):
            log.info(f"Train loop: {i}")
            train_stgcn(config, path + f"/{i}/")
    else:
        log.info(f"Model not found ({args.model })!")

    log.info(f"Training done. Output path: {path}")


elif args.eval:
    torch.cuda.empty_cache()

    name = args.name
    config = {}

    #### Load Configuration ####
    if args.config is not None:
        if ".yml" in args.config or ".yaml" in args.config:
            with open(args.config) as file:
                name = args.name
                config = yaml.load(file, Loader=yaml.FullLoader)
        elif ".json" in args.config:
            with open(args.config) as file:
                name = args.name
                config = json.load(file)
        else:
            with open("./experiments/config_repo.yml") as file:
                name += f"_{args.config}"
                config = yaml.load(file, Loader=yaml.FullLoader)[args.config]
    else:
        with open("./experiments/config_repo.yml") as file:
            name += "_standard"
            config = yaml.load(file, Loader=yaml.FullLoader)["standard"]

    exp = Experiment(**config["exp"])

    if "mode" in config and config["mode"] == "sklearn":
        exp.evaluate_emb()


elif args.prep_data:

    data_paths = [
        "D:/Temporal Info Graph/kinetics-skeleton/kinetics_train/",
        "D:/Temporal Info Graph/kinetics-skeleton/kinetics_val/"
    ]

    output = "C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/"

    log.info("Process data - Class information")

    _ = TIGDataset(name="kinetics-skeleton", s_files=data_paths, path=output,
                   verbose=True, process_label=True)

    log.info("Data set processed")


elif args.baseline:
    log.info(f"Run baseline on {args.data}")
    data = TIGDataset(name=args.data, path="../content/")
    data.x = data.x.reshape(data.x.shape[0], -1)


    top5 = []
    top1 = []
    print("Start baseline training")    
    for i in trange(10):
        model = get_model("svm")
        model.fit(data.x, data.y)

        yhat = model.decision_function(data.x)                
        # accuracy_score(emb_y, yhat)
        top5.append(top_k_accuracy_score(data.y, yhat, k = 5))
        top1.append(top_k_accuracy_score(data.y, yhat, k = 1))
        print(f"Iteration {i} | Top1: {np.mean(top1)} Top5: {np.mean(top5)} ")

    print("Save baseline accuracies...")
    np.save("../content/top1_svm.npy", top1)
    np.save("../content/top5_svm.npy", top5)

