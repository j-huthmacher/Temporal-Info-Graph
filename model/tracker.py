"""
    @author: jhuthmacher
"""
import io
from typing import Any

import torch
import pymongo
import gridfs

from sacred import Experiment
from sacred.observers import MongoObserver

class Tracker():
    
    def __init__(self, ex_name = None, db_url = None, db="temporal_info_graph", interactive=False, config = {}):
        """
        """
        if ex_name is not None:
            self.ex = Experiment(ex_name, interactive= interactive)
            self.ex.observers.append(MongoObserver(url=db_url, db_name=db))
            self.observer = self.ex.observers[0]
            self.run_collection = self.ex.observers[0].runs

            self.id = None
            
            # Assign model config
            self.__dict__ = {**self.__dict__, **config}
            self.run = None
          
    def track(self, mode):
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
            self.ex.main(mode)
            self.ex.run()
            
            #############
            # After run #
            #############

            # Important values in config get overwritten!!! I.e. track after run is done
            self.log_config("optimzer", str(self.solver.optimizer))

            buffer = io.BytesIO()
            torch.save(self.solver.model, buffer)

            self.add_artifact(buffer.getvalue(), name="model")


        elif mode == "epoch":
            self.track_epoch()
        elif mode == "training":
            self.track_train()
        elif mode == "validation":
            self.track_validation()

    def track_epoch(self):
        """ Function that manage the tracking per epoch (called from the solver).
        """
        self.ex.log_scalar("loss.epoch.train", self.solver.train_losses[self.solver.epoch], self.solver.epoch)
        self.ex.log_scalar("loss.epoch.val", self.solver.val_losses[self.solver.epoch], self.solver.epoch)

    def track_train(self):
        """ Function that manage the tracking per trainings batch (called from the solver).
        """
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return

        self.ex.log_scalar(f"loss.batch.train.{self.solver.epoch}", self.solver.train_batch_losses[self.solver.batch], self.solver.batch)
    
    def track_validation(self):
        """ Function that manage the tracking per validation batch (called from the solver).
        """
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return

        self.ex.log_scalar(f"loss.batch.val.{self.solver.epoch}", self.solver.val_batch_losses[self.solver.batch], self.solver.batch)

    def track_traning(self, train):
        """
        """
        def inner(*args):
            # Extract solver
            self.solver = train.__self__            
            train(*args, track = self.track)            
        
        return inner
    
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

        self.run_collection.update({'_id' : self.id}, {'$set' : {f'config.{name}':  cfg} })

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
