""" File containing different experiement configurations.
    @author: jhuthmacher
"""
import io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

from model import TemporalInfoGraph, MLP, Solver, Tracker
from data.tig_data_set import TIGDataset
from data import KINECT_ADJACENCY

from config.config import log

default = {
        "data": {
        },
        "data_split": {
        },
        "stratify": False,
        "loader": {
        },
        "same_loader": False,
        "emb_tracking": False,        
        "encoder": {
        },
        "encoder_training": {
        },
        "classifier": {
        },
        "classifier_training": {
        },
    }


def experiment(tracker, config):
    config = {**default, **config}
    def run(_run):
        log.info("Start experiment (Overfitting).")
        tracker.run = _run
        tracker.id = _run._id
        tracker.log_config(f"{tracker.tag}local_path", str(tracker.local_path))

        data = TIGDataset(**config["data"])

        if not config["stratify"] or config["stratify"] == {} :
            train, val = data.split(**config["data_split"]) # Default: 80% train, 10% val
        else:
            train = data.stratify(**config["stratify"])
            val = train # TODO: Implement stratify also for validation

        if "val_loader" in config["loader"]:
            train_loader = DataLoader(train, **config["loader"]["train_loader"])
            val_loader = DataLoader(val, **config["loader"]["val_loader"])
        else:
            # In this case we use the same data loader config for train and validation.
            train_loader = DataLoader(train, **config["loader"])
            val_loader = DataLoader(val, **config["loader"])

        if config["same_loader"]:
            loader = [train_loader, train_loader]
        else:
            loader = [train_loader, val_loader]

        tig = TemporalInfoGraph(**config["encoder"]).cuda()
        solver = Solver(tig, loader)

        #### Tracking ####
        tracker.track_traning(solver.train)(config["encoder_training"])

        if config["emb_tracking"]:
            emb_x = np.array([])
            emb_y = np.array([])
            for batch_x, batch_y in train_loader:
                pred, _ = solver.model(batch_x.type("torch.FloatTensor").permute(0,3,2,1),
                                       torch.tensor(KINECT_ADJACENCY))
                if emb_x.size:
                    emb_x = np.concatenate([emb_x, pred.detach().cpu().numpy()])
                else:
                    emb_x = pred.detach().cpu().numpy()

                if emb_y.size:
                    emb_y = np.concatenate([emb_y, batch_y.numpy()])
                else:
                    emb_y = batch_y.numpy()

            buffer = io.BytesIO()
            np.savez(buffer, x = emb_x, y= emb_y)
            tracker.add_artifact(buffer.getvalue(), name="embeddings.npz")

            #### Local Embedding Tracking ####
            np.savez(tracker.local_path+"embeddings", x = emb_x, y= emb_y)

        if not "classifier" in config and not isinstance(config["classifier"], dict):
            return

        #### DOWNSTREAM ####
        tracker.tag = "MLP."
        # Use the trained model!
        num_classes = config["classifier"]["num_classes"] if "num_classes" in config["classifier"] else int(np.max(data.y) + 1) 
        classifier = MLP(num_class=num_classes, encoder=solver.model, **config["classifier"]).cuda()

        solver = Solver(classifier, loader, loss_fn = nn.CrossEntropyLoss())

        tracker.track_traning(solver.train)(config["classifier_training"])
        # tracker.track_testing(solver.test)({"verbose": True })

        log.info("Experiment done.")

    return run












def exp_colab(tracker):
    def run(_run):
        log.info("Start experiment.")        
        tracker.run = _run
        tracker.id = _run._id

        tracker.save_nth = 25

        #### Data Set Up ####
        data = TIGDataset("kinetic_skeleton_5000", path="/content/")
        train, val = data.split() # Default: 80% train, 10% val

        train_loader = DataLoader(train, batch_size=4)
        val_loader = DataLoader(val, batch_size=4)
        # test_loader = DataLoader(test, batch_size=len([1]))

        #### Encoder ####
        if tracker.checkpoint is not None:
            log.info("Checkpoint exists")
            solver = Solver((tracker.checkpoint, TemporalInfoGraph), [train_loader, train_loader])
        else:
            tracker.checkpoint_dict["model_params"] = dict(c_in=4, c_out=16, spec_out=16,
                                                           out=16, dim_in=(18, 300), tempKernel=32)

            tig = TemporalInfoGraph(**tracker.checkpoint_dict["model_params"]).cuda()
            tracker.log_config(f"{tracker.tag}num_paramters", str(tig.num_paramters))
            solver = Solver(tig, [train_loader, val_loader])
        
        tracker.track_traning(solver.train)({
            "verbose": True, 
            "n_epochs": 1024
            })
        log.info("Encoder training done!")

        #### Downstream ####
        log.info("Starting downstream training...")
        tracker.tag = "MLP."
        # Use the trained model!
        num_classes = 400 #int(np.max(y) + 1)
        classifier = MLP(16, num_classes, [256, 512, 1024], tracker.solver.model).cuda()

        tracker.log_config(f"{tracker.tag}MLP.model", str(classifier))
        tracker.log_config(f"{tracker.tag}MLP.layers", str([64, 256, 512, num_classes]))

        solver = Solver(classifier, [train_loader, val_loader], loss_fn = nn.CrossEntropyLoss())

        tracker.track_traning(solver.train)({
            "verbose": True,
            "n_epochs": 1024,
            })

        log.info("Experiment done.")
    return run


def exp_test_trained_enc(tracker):
    def run(_run):
        log.info("Start experiment.")
        
        tracker.run = _run
        tracker.id = _run._id
        # tracker.run.experiment_info['name'] = "MLP_Learning_Overfitting_Behavior"

        # data = TIGDataset("kinetic_skeleton_5000", path="./dataset/")
        # x = data.x[:100]
        # y = data.y[:100]

        # tracker.log_config(f"{tracker.tag}label_distr", str(y))

        # # TODO: Stratified sampling
        # sampler = RandomOverSampler(random_state=0)
        # idx, _ = sampler.fit_sample(np.arange(y.shape[0]).reshape(-1, 1), y)

        # x = x[idx.squeeze()]
        # y = y[idx.squeeze()]

        data = TIGDataset("kinetic_skeleton_5000", path="/content/")
        train, val = data.split(lim=500) # Default: 80% train, 10% val

        train_loader = DataLoader(train, batch_size=32)
        val_loader = DataLoader(val, batch_size=32)


        # Replace class lables to lower values!
        # keys = np.unique(data[:, 1]) 
        # vals = np.arange(np.unique(data[:, 1]).shape[0])

        # for k,v in zip(keys, vals):
        #     data[data[:, 1] == k, 1] = v


        #### DOWNSTREAM ####
        log.info("Starting downstream training...")

        
        # Use the trained model!
        encoder = torch.load("./output/04122020_1641/TIG_.pt")
        num_classes = 400 #int(np.max(y) + 1)
        classifier = MLP(64, num_classes, [256, 512, 1024, 1024, 1024, 1024, 1024, 1024], encoder).cuda()

        tracker.log_config(f"{tracker.tag}MLP.model", str(classifier))
        tracker.log_config(f"{tracker.tag}MLP.layers", str([64, 256, 512, 1024, 1024, 1024, 1024, 1024, 1024, num_classes]))

        solver = Solver(classifier, [train_loader, train_loader], loss_fn = nn.CrossEntropyLoss())

        train_cfg = {
            "verbose": True,
            "n_epochs": 5000,
            "learning_rate": 2,
            "weight_decay": None
        }

        tracker.track_traning(solver.train)(train_cfg)

        # tracker.log_config(f"{tracker.tag}optimzer", "Trained encoder (500 samples)")

        log.info("Experiment done.")
    return run


def exp_test(tracker):
    def run(_run):
        log.info("Start experiment.")
        tracker.run = _run
        tracker.id = _run._id

        data = TIGDataset("kinetic_skeleton_5000", path="/content/")
        train, val = data.split(lim=100) # Default: 80% train, 10% val

        train_loader = DataLoader(train, batch_size=4)
        val_loader = DataLoader(val, batch_size=4)

        tig = TemporalInfoGraph(c_in=4, c_out=6, spec_out=7, out=64, dim_in=(18, 300), tempKernel=32).cuda()
        solver = Solver(tig, [train_loader, val_loader])  

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 500 })

        #### DOWNSTREAM ####
        log.info("Starting downstream training...")
        tracker.tag = "MLP."
        # Use the trained model!
        num_classes = np.max(data[:, 1]) + 1
        classifier = MLP(64, num_classes, [128, 512, 256], tracker.solver.model).cuda()

        solver = Solver(classifier, [train_loader, val_loader, test_loader], loss_fn = nn.CrossEntropyLoss())

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 500 })
        # tracker.track_testing(solver.test)({"verbose": True, "n_epochs": 1 })

        log.info("Experiment done.")

    return run


def exp_tig_overfit(tracker):
    def run(_run):
        log.info("Start experiment (Overfitting).")
        tracker.run = _run
        tracker.id = _run._id
        tracker.log_config(f"{tracker.tag}local_path", str(tracker.local_path))

        data = TIGDataset("kinetic_skeleton_5000", path="/content/")
        train, val = data.split(lim=500) # Default: 80% train, 10% val

        train_loader = DataLoader(train, batch_size=32)
        val_loader = DataLoader(val, batch_size=32)

        if tracker.checkpoint is not None:
            log.info("Checkpoint exists")
            solver = Solver((tracker.checkpoint, TemporalInfoGraph), [train_loader, train_loader])
        else:
            tracker.checkpoint_dict["model_params"] = dict(c_in=4, c_out=64, spec_out=128,
                                                           out=256, dim_in=(18, 300), tempKernel=32)
            # tracker.checkpoint["optim_params"] = dict(c_in=4, c_out=64, spec_out=128, out=256, dim_in=(18, 300), tempKernel=32)

            tig = TemporalInfoGraph(**tracker.checkpoint_dict["model_params"]).cuda()
            solver = Solver(tig, [train_loader, train_loader])

        #### Tracking ####
        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 512 })

        log.info("Experiment done.")

    return run


def exp_overfit(tracker):
    def run(_run):
        log.info("Start experiment (Overfitting).")
        tracker.run = _run
        tracker.id = _run._id
        tracker.log_config(f"{tracker.tag}local_path", str(tracker.local_path))

        data = TIGDataset("kinetic_skeleton_5000", path="./content/")
        # train, val = data.split(lim=500) # Default: 80% train, 10% val

        train = data.stratify(5)

        train_loader = DataLoader(train, batch_size=32)
        # val_loader = DataLoader(val, batch_size=32)

        checkpoint_dict = dict(c_in=2, c_out=64, spec_out=128,
                               out=2, dim_in=(36, 300), tempKernel=32)

        tig = TemporalInfoGraph(**checkpoint_dict).cuda()
        solver = Solver(tig, [train_loader, train_loader])

        #### Tracking ####
        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 100 })
        log.info("Experiment done.")

        emb_x = np.array([])
        emb_y = np.array([])
        for batch_x, batch_y in train_loader:
            pred, _ = solver.model(batch_x.type("torch.FloatTensor").permute(0,3,2,1), torch.tensor(KINECT_ADJACENCY))
            if emb_x.size:
                emb_x = np.concatenate([emb_x, pred.detach().cpu().numpy()])
            else:
                emb_x = pred.detach().cpu().numpy()
            
            if emb_y.size:
                emb_y = np.concatenate([emb_y, batch_y.numpy()])
            else:
                emb_y = batch_y.numpy()

        np.savez(tracker.local_path+"embeddings", x = emb_x, y= emb_y)

        return

        #### DOWNSTREAM ####
        tracker.tag = "MLP."
        # Use the trained model!
        num_classes = int(np.max(data.y) + 1)
        classifier = MLP(4, num_classes, [128, 512, 1024, 256], solver.model).cuda()

        solver = Solver(classifier, [train_loader, train_loader], loss_fn = nn.CrossEntropyLoss())

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 512 })
        # tracker.track_testing(solver.test)({"verbose": True })

        log.info("Experiment done.")

    return run


####################
# Helper functions #
####################

def coll(batch):
    # x = np.asarray(np.asarray(batch)[:, 0].tolist())
    # y = np.asarray(np.asarray(batch)[:, 1].tolist())

    x = np.asarray(batch[0])
    y = np.asarray(batch[1])

    return (x, y)