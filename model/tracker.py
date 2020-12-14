"""
    @author: jhuthmacher
"""
import io
import os
from typing import Any
from pathlib import Path
from datetime import datetime
import json
import torch
import pymongo
import gridfs
import numpy as np
import matplotlib.pyplot as plt

from sacred import Experiment
from sacred.observers import MongoObserver

from model import Solver, MLP, TemporalInfoGraph
from config.config import log
from visualization import class_contour, create_gif, plot_desc_loss_acc
from data import KINECT_ADJACENCY


class Tracker(object):
    
    def __init__(self, ex_name = None, db_url = None, db="temporal_info_graph", interactive=False, config = {},
                 local = True, local_path: str = None, track_decision = False):
        """
        """
        super().__init__()

        if ex_name is not None and db_url is not None:
            client = pymongo.MongoClient(db_url)

            self.ex_name = ex_name
            self.ex = Experiment(ex_name, interactive=interactive)
            self.ex.logger = log
            self.ex.observers.append(MongoObserver(client=client, db_name=db))
            self.observer = self.ex.observers[0]
            self.run_collection = self.ex.observers[0].runs

            self.db_url = db_url
            self.db = client[db]
            self.interactive = interactive
            self.id = None
            self.run = None
        else:
            self.ex = None
        
        self.tag = ""
        self.local = local
        self.track_decision = track_decision

        self.date = datetime.now().strftime("%d%m%Y_%H%M")

        self.checkpoint_dict = {
                "model_params": {},
                "optim_params": {},
            }

        self.local_path = f"./output/{self.date}_{ex_name}/" if local_path is None else local_path
        Path(self.local_path).mkdir(parents=True, exist_ok=True)

        self.save_nth = 100
                                        
        # Assign model config
        self.__dict__ = {**self.__dict__, **config}
        

        self.cfg = {}


    
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
            self.cfg = cfg
            if self.ex is not None:
                # Mode corresponds to an function       
                self.ex.main(mode(self, cfg))
                self.ex.run()
            else:
                mode(self, cfg)()
            
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
        try:
            if isinstance(self.solver.model, MLP) and self.track_decision:
                emb_x = np.array(self.solver.train_loader.dataset, dtype=object)[:, 0]
                emb_y = np.array(self.solver.train_loader.dataset, dtype=object)[:, 1]

                loss = {
                    "MLP Train Loss": self.solver.train_losses,
                    "MLP Val Loss": self.solver.train_losses,
                }

                metric = {
                    "MLP Top-1 Acc.": np.array(self.solver.train_metrics)[:, 0],
                    "MLP Top-5 Acc.": np.array(self.solver.train_metrics)[:, 1],
                    "MLP Val Top-1 Acc.": np.array(self.solver.val_metrics)[:, 0],
                    "MLP Val Top-5 Acc.": np.array(self.solver.val_metrics)[:, 1],
                }

                
                fig = plot_desc_loss_acc(emb_x,
                                        emb_y,
                                        self.solver.model,
                                        loss = loss,
                                        metric=metric,
                                        n_epochs = self.solver.train_cfg["n_epochs"],
                                        title=f"MLP Decision Boundary - Epoch: {self.solver.epoch}",
                                        model_name="MLP")
                plt.close()
                create_gif(fig, path=self.local_path+"MLP.decision.boundaries.gif",
                        fill=(self.solver.train_cfg["n_epochs"] - 1 != self.solver.epoch))

            if isinstance(self.solver.model, TemporalInfoGraph) and self.track_decision:
                with torch.no_grad():
                    emb_x = np.array(list(np.array(self.solver.train_loader.dataset, dtype=object)[:, 0]))
                    emb_x = self.model(emb_x.transpose((0,3,2,1)), KINECT_ADJACENCY)[0]
                emb_y = np.array(self.solver.train_loader.dataset, dtype=object)[:, 1]

                loss = {
                    "TIG Train Loss (JSD MI)": self.solver.train_losses,
                    "TIG Val Loss (JSD MI)": self.solver.train_losses,
                }

                fig = plot_desc_loss_acc(emb_x,
                                        emb_y,
                                        None,
                                        loss,
                                        None,
                                        n_epochs = self.solver.train_cfg["n_epochs"],
                                        title=f"TIG Embeddings (PCA) - Epoch: {self.solver.epoch}",
                                        model_name="TIG",
                                        mode="PCA")
                if self.solver.epoch == 0:
                    fig.savefig(self.local_path+"TIG.embeddings.pca.initial.png")

                plt.close()
                create_gif(fig, path=self.local_path+"TIG.embeddings.pca.gif",
                        fill=(self.solver.train_cfg["n_epochs"] - 1 != self.solver.epoch))
                
                fig = plot_desc_loss_acc(emb_x,
                                        emb_y,
                                        None,
                                        loss,
                                        None,
                                        n_epochs = self.solver.train_cfg["n_epochs"],
                                        title=f"TIG Embeddings (TSNE) - Epoch: {self.solver.epoch}",
                                        model_name="TIG",
                                        mode="TSNE")
                if self.solver.epoch == 0:
                    fig.savefig(self.local_path+"TIG.embeddings.tsne.initial.png")

                plt.close()
                create_gif(fig, path=self.local_path+"TIG.embeddings.tsne.gif",
                        fill=(self.solver.train_cfg["n_epochs"] - 1 != self.solver.epoch))
        except:
            pass

        
        if self.ex is not None:        
            if self.solver.epoch % self.save_nth == 0:
                self.track_checkpoint() 

            self.ex.log_scalar(f"{self.tag}loss.epoch.train", self.solver.train_losses[self.solver.epoch], self.solver.epoch)
            if hasattr(self.solver, "train_metric"):
                self.ex.log_scalar(f"{self.tag}train.top1", self.solver.train_metric[0])
                self.ex.log_scalar(f"{self.tag}train.top5", self.solver.train_metric[1])

        if self.solver.phase == "validation" and self.ex is not None:
            self.ex.log_scalar(f"{self.tag}loss.epoch.val", self.solver.val_losses[self.solver.epoch], self.solver.epoch)

            if hasattr(self.solver, "val_metric"):
                self.ex.log_scalar(f"{self.tag}val.top1", self.solver.val_metric[0])
                self.ex.log_scalar(f"{self.tag}val.top5", self.solver.val_metric[1])
            
    def track_train(self):
        """ Function that manage the tracking per trainings batch (called from the solver).
        """
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return
        if self.ex is not None: 
            self.ex.log_scalar(f"{self.tag}.loss.batch.train.{self.solver.epoch}",
                               self.solver.train_batch_losses[self.solver.batch], self.solver.batch)
    
    def track_validation(self):
        """ Function that manage the tracking per validation batch (called from the solver).
        """
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return
        if self.ex is not None: 
            self.ex.log_scalar(f"{self.tag}.loss.batch.val.{self.solver.epoch}",
                               self.solver.val_batch_losses[self.solver.batch], self.solver.batch)
    
    def track_evaluation(self):
        """ Function that manage the tracking per validation batch (called from the solver).
        """
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return
        if self.ex is not None: 
            self.ex.log_scalar(f"{self.tag}test.top1", self.solver.metric[0])
            self.ex.log_scalar(f"{self.tag}test.top5", self.solver.metric[1])

    def track_traning(self, train):
        """
        """
        def inner(*args):
            # Extract solver
            self.solver = train.__self__            
            train(*args, track = self.track)
            
            if self.ex is not None: 
                #### LOGGING ####
                
                self.log_config(f"{self.tag}optimzer", str(self.solver.optimizer))
                self.log_config(f"{self.tag}train_cfg", str(self.solver.train_cfg))  
                self.log_config(f"{self.tag}train_size", str(len(self.solver.train_loader.dataset)))
                self.log_config(f"{self.tag}train_batch_size", str(self.solver.train_loader.batch_size))
                self.log_config(f"{self.tag}model", str(self.solver.model))

                buffer = io.BytesIO()
                torch.save(self.solver.model, buffer)

                self.add_artifact(buffer.getvalue(), name=f"{self.ex_name}.pt")

            if self.solver.phase == "validation":
                self.log_config(f"{self.tag}val_size", str(len(self.solver.val_loader.dataset)))
                self.log_config(f"{self.tag}val_batch_size", str(self.solver.val_loader.batch_size))

            self.track_locally()
               
        return inner
    
    def track_testing(self, test):
        def inner(*args):
            # Extract solver
            self.solver = test.__self__            
            test(*args, track = self.track)  

            if self.ex is not None: 
                if self.solver.test_loader is not None:
                    self.log_config(f"{self.tag}test_cfg", str(self.solver.test_cfg))
                    self.log_config(f"{self.tag}test_size", str(np.array(self.solver.test_loader.dataset).shape))
                    self.log_config(f"{self.tag}test_batch_size", str(self.solver.test_loader.batch_size))

                self.log_config(f"{self.tag}model", str(self.solver.model))          
        
        return inner

    def track_locally(self):
        """
        """
        Path(self.local_path).mkdir(parents=True, exist_ok=True)

        with open(f'{self.local_path}/config.json', 'w') as fp:
            json.dump({
                **self.cfg, 
                **{ 
                    "val_data_size": len(self.solver.val_loader.dataset) if hasattr(self.solver, "val_loader") else "-",
                    "train_data_size": len(self.solver.train_loader.dataset),
                    "exec_dir": os.getcwd(),
                    "optimzer": str(self.solver.optimizer),
                    }}, fp)

        np.save(f'{self.local_path}/TIG_{self.tag}train_losses.npy', self.solver.train_losses)
        
        if hasattr(self.solver, "train_metric"):
            np.save(f"{self.local_path}/TIG_{self.tag}train.metrics.npy", self.solver.train_metrics)


        np.save(f'{self.local_path}/TIG_{self.tag}val_losses.npy', self.solver.val_losses)

        if hasattr(self.solver, "val_metric"):
            np.save(f"{self.local_path}/TIG_{self.tag}val.metrics.npy", self.solver.val_metrics)
        
        #### Test / Evaluation Metrics ####
        if hasattr(self.solver, "metric"):
            np.save(f'{self.local_path}/TIG_{self.tag}top1.npy', self.solver.metric[0])
            np.save(f'{self.local_path}/TIG_{self.tag}top5.npy', self.solver.metric[1])

        torch.save(self.model, f'{self.local_path}/TIG_{self.tag.replace(".", "")}.pt')
        log.info(f"Experiment stored at '{self.local_path}")

    def track_checkpoint(self):
        """
        """
        self.checkpoint_dict = {
            **self.checkpoint_dict, 
            **{
                'epoch': self.solver.epoch,
                'model_state_dict': self.solver.model.state_dict(),
                'optimizer_state_dict': self.solver.optimizer.state_dict(),
                'loss': self.solver.val_losses[-1] if len(self.solver.val_losses) > 0 else "-",
            }
           }
        
        #### Remote ####
        buffer = io.BytesIO()
        torch.save(self.checkpoint, buffer)

        self.add_artifact(buffer.getvalue(), name="checkpoint.pt")

        #### Local Checkpoint ####
        path = self.local_path + "checkpoints/"
        Path(path).mkdir(parents=True, exist_ok=True)
        
        torch.save(self.checkpoint_dict, f"{path}checkpoint.pt")

        log.info("Checkpoint stored")


    @property
    def checkpoint(self):
        """
        """
        try:
            return torch.load(f"{self.local_path}checkpoints/checkpoint.pt")
        except Exception as e:
            # Checkpoint doesn't exists
            return None



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
        if self.ex is None:
            return
        
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
        if self.ex is None:
            return
    
        db_filename = "artifact://{}/{}/{}".format(self.observer.runs.name, self.id, name)
        
        result = self.db["fs.files"].find_one({"name": self.observer.fs.delete(db_filename)})
        if "_id" in result:
            self.observer.fs.delete(result['_id'])
        
        file_id = self.observer.fs.put(file, filename=db_filename)

        self.observer.run_entry["artifacts"].append({"name": name, "file_id": file_id})
        self.observer.save()
