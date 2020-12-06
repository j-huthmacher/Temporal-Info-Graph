"""
    @author: jhuthmacher
"""
import io
from typing import Any
from pathlib import Path
from datetime import datetime

import torch
import pymongo
import gridfs
import numpy as np

from sacred import Experiment
from sacred.observers import MongoObserver

from model.solver import Solver
from config.config import log


class Tracker():
    
    def __init__(self, ex_name = None, db_url = None, db="temporal_info_graph", interactive=False, config = {},
                 local = True):
        """
        """
        if ex_name is not None:
            self.ex_name = ex_name
            self.ex = Experiment(ex_name, interactive=interactive)
            self.ex.logger = log
            self.ex.observers.append(MongoObserver(url=db_url, db_name=db))
            self.observer = self.ex.observers[0]
            self.run_collection = self.ex.observers[0].runs

            self.db_url = db_url
            self.db = db
            self.interactive = interactive
            self.id = None
            self.tag = ""
            self.local = local

            self.date = datetime.now().strftime("%d%m%Y_%H%M")
                            
            # Assign model config
            self.__dict__ = {**self.__dict__, **config}
            self.run = None
    
    @property
    def model(self):
        # TODO: Catch if the model/solver doesn't exists yet.
        return self.solver.model
    
    def new_ex(self, ex_name):
        """
        """
        self.ex_name = ex_name
        self.ex = Experiment(ex_name, interactive=self.interactive)
        self.ex.observers.append(MongoObserver(url=self.db_url, db_name=self.db))
        self.observer = self.ex.observers[0]
        self.run_collection = self.ex.observers[0].runs
              
    def track(self, mode: str, cfg = None):
        """ General track function that distributes the task depending on the mode.

            Paramters:
                mode: Callable or str
                    Mode can be either a function or a string. In case of an function the 
                    track function starts the experiment with the handovered function as main.
                    If it is a string the experiment is already started and the function delegate 
                    the tracking task to the corresponding function.
        """
        if callable(mode):
            # Mode corresponds to an function            
            self.ex.main(mode(self))
            self.ex.run()
            
            #############
            # After run #
            #############

            # Important values in config get overwritten!!! I.e. track after run is done
            # self.log_config(f"{self.tag}optimzer", str(self.solver.optimizer))
            # self.log_config(f"{self.tag}train_cfg", str(self.solver.train_cfg))
            # self.log_config(f"{self.tag}test_cfg", str(self.solver.test_cfg))
            # Important values in config get overwritten!!! I.e. track after run is done

            if self.local:
                self.track_locally()     
            

        elif mode == "epoch":
            self.track_epoch()            
        elif mode == "training":            
            self.track_train()            
        elif mode == "validation":
            self.track_validation()            
        elif mode == "evaluation":
            self.track_evaluation()
            

    def track_epoch(self):
        """ Function that manage the tracking per epoch (called from the solver).
        """
        self.ex.log_scalar(f"{self.tag}loss.epoch.train", self.solver.train_losses[self.solver.epoch], self.solver.epoch)
        self.ex.log_scalar(f"{self.tag}loss.epoch.val", self.solver.val_losses[self.solver.epoch], self.solver.epoch)

        if hasattr(self.solver, "val_metric"):
            self.ex.log_scalar(f"{self.tag}val.top1", self.solver.val_metric[0])
            self.ex.log_scalar(f"{self.tag}val.top5", self.solver.val_metric[1])

    def track_train(self):
        """ Function that manage the tracking per trainings batch (called from the solver).
        """
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return

        self.ex.log_scalar(f"{self.tag}.loss.batch.train.{self.solver.epoch}", self.solver.train_batch_losses[self.solver.batch], self.solver.batch)
    
    def track_validation(self):
        """ Function that manage the tracking per validation batch (called from the solver).
        """
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return

        self.ex.log_scalar(f"{self.tag}.loss.batch.val.{self.solver.epoch}", self.solver.val_batch_losses[self.solver.batch], self.solver.batch)
    
    def track_evaluation(self):
        """ Function that manage the tracking per validation batch (called from the solver).
        """
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return

        self.ex.log_scalar(f"{self.tag}test.top1", self.solver.metric[0])
        self.ex.log_scalar(f"{self.tag}test.top5", self.solver.metric[1])

    def track_traning(self, train):
        """
        """
        def inner(*args):
            # Extract solver
            self.solver = train.__self__            
            train(*args, track = self.track)

            #### LOGGING ####
            self.log_config(f"{self.tag}optimzer", str(self.solver.optimizer))
            self.log_config(f"{self.tag}train_cfg", str(self.solver.train_cfg))  
            self.log_config(f"{self.tag}train_size", str(len(self.solver.train_loader.dataset)))
            self.log_config(f"{self.tag}train_batch_size", str(self.solver.train_loader.batch_size))

            if self.solver.val_loader is not None:
                self.log_config(f"{self.tag}val_size", str(len(self.solver.val_loader.dataset)))
                self.log_config(f"{self.tag}val_batch_size", str(self.solver.val_loader.batch_size))

            self.log_config(f"{self.tag}model", str(self.solver.model))

            buffer = io.BytesIO()
            torch.save(self.solver.model, buffer)

            self.add_artifact(buffer.getvalue(), name=f"{self.ex_name}.pt")

            self.track_locally()
               
        return inner
    
    def track_testing(self, test):
        def inner(*args):
            # Extract solver
            self.solver = test.__self__            
            test(*args, track = self.track)  

            if self.solver.test_loader is not None:
                self.log_config(f"{self.tag}test_cfg", str(self.solver.test_cfg))
                self.log_config(f"{self.tag}test_size", str(len(self.solver.test_loader.dataset)))
                self.log_config(f"{self.tag}test_batch_size", str(self.solver.test_loader.batch_size))

            self.log_config(f"{self.tag}model", str(self.solver.model))          
        
        return inner

    def track_locally(self):
        """
        """
        Path(f"./output/{self.date}/").mkdir(parents=True, exist_ok=True)

        np.save(f'./output/{self.date}/TIG_{self.tag}_train_losses.npy', self.solver.train_losses)
        np.save(f'./output/{self.date}/TIG_{self.tag}_val_losses.npy', self.solver.val_losses)
        
        if hasattr(self.solver, "metric"):
            np.save(f'./output/{self.date}/TIG_{self.tag}_top1.npy', self.solver.metric[0])
            np.save(f'./output/{self.date}/TIG_{self.tag}_top5.npy', self.solver.metric[1])

        torch.save(self.model, f'./output/{self.date}/TIG_{self.tag}.pt')
        log.info(f"Experiment stored at './output/{self.date}/")

    ##########################
    # Custom Sacred Tracking #
    ##########################

    def load_model(self, run_id: int, name: str, db_url: Any = None, mode: str = "torch"):
        """ Function to load a model from the tracking server.

            You can either use the configuration (e.g. DB connection) from the tracker,
            if it is initialized or you provide the DB URL to create a new connection
            to the DB.

            Paramters:
                run_id: int
                    ID of the run from which you want to load the model.
                name: str
                    File name of the model
                db_url: str
                    DB url that is used to establish a new connection to the DB.
                mode: str
                    To determine if the loaded model is returned as PyTorch object or 
                    as a plain bytes object.
            Return:
                torch.nn.Module or bytes object.
        """

        if hasattr(self, "observer") and self.observer is not None and db_url is None:
            runs = self.run_collection
            fs = self.observer.fs
        else:
            print(db_url)
            client = pymongo.MongoClient(db_url)
            fs = gridfs.GridFS(client.temporal_info_graph)
            runs = client.temporal_info_graph.runs

        run_entry = runs.find_one({'_id': run_id, "artifacts": { "$elemMatch": { "name": name } } })

        if mode == "torch":
            buff = fs.get(run_entry['artifacts'][0]['file_id']).read()
            buff = io.BytesIO(buff)
            buff.seek(0)
            return torch.load(buff)
        else:
            return fs.get(run_entry['artifacts'][0]['file_id']).read()
    
    def log_config(self, name: str, cfg: object):
        """ Manually track configuration to data base.

            Paramters:
                name: str
                    Name of the configuration that will be tracked.
                cfg: object
                    The configuration that should be stored in the DB.
                    For instance, a dictionary or a simple string.
        """
        if self.id is None:
            raise ValueError("Experiment ID is not set!")
        
        # self.run_collection.update({'_id' : self.id}, {'$set' : {f'config.{name}':  cfg} })

        self.observer.run_entry["config"][name] = cfg
        self.observer.save()

    def add_artifact(self, file: bytes, name: str):
        """ Tracks an object.

            Paramters:
                file: bytes
                    File that should be tracked as artifact.
                name: str
                    Name of the artifact.
        """

        db_filename = "artifact://{}/{}/{}".format(self.observer.runs.name, self.id, name)
        file_id = self.observer.fs.put( file, filename=db_filename)

        self.observer.run_entry["artifacts"].append({"name": name, "file_id": file_id})
        self.observer.save()
