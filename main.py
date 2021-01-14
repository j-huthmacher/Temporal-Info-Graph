#pylint: disable=line-too-long
""" Main file.

    @author: jhuthmacher
"""
import json

import argparse
import torch
import yaml

#pylint: disable=import-error
from tracker import Tracker
from experiments import  experiment


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
parser.add_argument('--downstream', dest='downstream', action='store_true',
                    help='Flag to determine if the downstream training should be executed.')
parser.add_argument('--disable_local_store', dest='disable_local_store', action='store_true',
                    help='Flag to determine if the models should be locally stored. Default: Models are stored locally.')
#### Data CLI ####
parser.add_argument('--prep_data', dest='prep_data', action="store_true",
                    help='Prepare data.')

# parser.add_argument('--type', dest='type', action="str",
#                     help='Prepare data.')

args = parser.parse_args()

#### Execution ####
if args.train:
    db_url = open(".mongoURL", "r").readline()
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

    #### Tracker Set Up ####
    if "name" in config:
        name += f"_{config['name']}"
    tracking = {"ex_name": name}
    if args.tracking == "remote":
        tracking = {
            "ex_name": name,
            "db_url": db_url,
            "interactive": True
        }

    # Training is executed from here
    if "tracking" in config:
        tracker = Tracker(**{**tracking, **config["tracking"]})
    else:
        tracker = Tracker(**tracking)

    # For reproducibility
    torch.manual_seed(0)
    config["seed"] = 0

    # experiment is the template function that is executed by the tracker and configured by "config"
    tracker.track(experiment, config)


elif args.prep_data:
    pass
    # Prepare data 
    # data_paths = [
    #     "C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/kinetics_train/",
    #     "C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/kinetics_val/"
    # ]

    # output = "C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/"

    # log.info("Process data")

    # _ = TIGDataset(paths = data_paths, output=output, verbose=True)

    # log.info("Data set processed")
