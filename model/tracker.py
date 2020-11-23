"""
    @author: jhuthmacher
"""
import io

import torch

from sacred import Experiment
from sacred.observers import MongoObserver

class Tracker():
    
    def __init__(self, ex_name, db_url, db="temporal_info_graph", interactive=False, config = {}):
        """
        """
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
            
            # After run!
            # Important values in config get overwritten!!! I.e. track after run is done
            self.log_config("optimzer", str(self.solver.optimizer))

            buffer = io.BytesIO()
            torch.save(self.solver.model, buffer)
            # buffer.seek(0)

            print(buffer)

            self.add_artifact(buffer.getvalue(), name="model")

            # self.ex.add_artifact(buffer.getvalue(), name="model")


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
    

    def log_config(self, name, cfg):
        """ Manually track configuration to data base.
        """
        if self.id is None:
            raise ValueError("Experiment ID is not set!")

        self.run_collection.update({'_id' : self.id}, 
                     {'$set' : {f'config.{name}':  cfg} })
    
    def add_artifact(self, file: bytes, name: str):
        """ Tracks an object.
        """

        db_filename = "artifact://{}/{}/{}".format(self.observer.runs.name, self.id, name)
        file_id = self.observer.fs.put( file, filename=db_filename)

        self.observer.run_entry["artifacts"].append({"name": name, "file_id": file_id})
        self.observer.save()
