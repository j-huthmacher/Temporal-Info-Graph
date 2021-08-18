# pylint: disable=line-too-long
""" Main file.

    Reflects the entry point for the TIG framework and provides the CLI.
"""
import json
from pathlib import Path
from datetime import datetime
import argparse

import yaml
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from tqdm import trange
import torch

from utils.tig_data_set import TIGDataset
from baseline import get_model
from processor import train_tig, train_stgcn
from config.config import create_logger
from evaluation import evaluate_experiments

#### Set Up CLI ####
parser = argparse.ArgumentParser(prog='tig', description='Temporal Info Graph')

#### Model/Experiment CLI ####
parser.add_argument('--config', dest='config',
                    help=('Defines which configuration should be usd. Can be a name,'
                          ' .json or .yml file. Depending on wich execution mode is selected the '
                          'config belongs to train config or evaluation config.'))
parser.add_argument('--name', dest='name', default="TIG_Experiment",
                    help='Name of the experiment.')
parser.add_argument('--tracking', dest='tracking', default="remote",
                    help='[remote, local], default: remote')
parser.add_argument('--train', dest='train', action='store_true',
                    help='Flag to select trainings mode.')
parser.add_argument('--model', dest='model', default="tig",
                    help='[tig, stgcn]')
parser.add_argument('--eval', dest='eval',
                    help=('Option to execute evaluation mode. It requires the path to the root of'
                          'the experiment as argument like `--eval path/to/experiment`'))
parser.add_argument('--model_name', dest='model_name', default="Model",
                    help='Name of the model that is displayed in the evaluation output.')

#### Data CLI ####
parser.add_argument('--prep_data', dest='prep_data', action="store_true",
                    help='Prepare data.')


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
        with open("./config/config.yml" if args.train else "./config/config_eval.yml") as file:
            name += f"_{args.config}"
            config = yaml.load(file, Loader=yaml.FullLoader)[args.config]
else:
    print("No config provided.")

#### Execute Training ####
if args.train:
    if "name" in config:
        name += f"_{config['name']}"  # Only for readability

    # For reproducibility
    torch.manual_seed(0)
    config["seed"] = 0

    #### Tracking Location ####
    date = datetime.now().strftime("%d%m%Y_%H%M")
    path_init = f"./output/{date}_{name}/"
    Path(path_init).mkdir(parents=True, exist_ok=True)

    log = create_logger(path_init)

    # To directly test different epochs we use an list of epochs that are subsequently executed.
    epoch_list = config["training"]["n_epochs"]

    for epochs in epoch_list:
        log.info("Epoch: %s", str(epochs))
        # For each epoch create an own tracking folder.
        Path(path_init + f"{epochs}epochs").mkdir(parents=True, exist_ok=True)

        config["training"]["n_epochs"] = epochs

        repetitions = 5
        if "repetitions" in config:
            repetitions = config["repetitions"]

        # Each experiment is repeated five times for more meaningful results
        for i in range(repetitions):
            log.info("Train loop: %d", i)
            path = path_init + f"{epochs}epochs/{i}/"
            Path(path).mkdir(parents=True, exist_ok=True)

            # Depending on the model different training methods are needed 
            if args.model == "tig":
                train_tig(config, path)
            elif args.model == "stgcn":
                train_stgcn(config, path)
            else:
                log.info("Model not found (%s)!", args.model)

    log.info("Training done. Output path: %s", str(path))

#### Execute Evaluation ####
elif args.eval:
    results = evaluate_experiments([(args.model_name, args.eval)], **config)

    results.to_csv(f"{args.eval}/evaluation_{datetime.now().strftime('%d%m%Y_%H%M')}.csv")

    print(f"Results are stored at: {args.eval}evaluation_{datetime.now().strftime('%d%m%Y_%H%M')}.csv")
    print(results[["Model", "Top-1 Accuracy", "Top-1 Std.", "Top-5 Accuracy", "Top-5 Std."]])


elif args.prep_data:
    # Was used to prepare the loosefiles from the kinetics skeleton data set.

    # Change paths!
    data_paths = [
        "local/path/to/kinetics-skeleton/kinetics_train/",
        "local/path/to/kinetics-skeleton/kinetics_val/"
    ]

    # Define output location
    output = "./"

    log.info("Process data")

    _ = TIGDataset(name="kinetics-skeleton", s_files=data_paths, path=output,
                   verbose=True, process_label=True)

    log.info("Data set processed")
