"""
"""
import unittest

import torch

from tracker.tracker import Tracker
from experiments import  experiment

# For reproducibility
torch.manual_seed(0)


class TestExperiment(unittest.TestCase):
    """ Test class for evaluation function.
    """

    def test_experiment(self):
        """
        """
        config = {
            "data": {
                "name": "kinetic_skeleton_5000",
                "path": "./content/",
                },
            "data_split":{
                "lim": 10
            },
            "loader": {
                "batch_size": 64
                },
            "encoder": {
                "architecture": [
                    [2,    32,  64,  64, 16],
                ]
            },
            "encoder_training": {
                "verbose": True,
                "n_epochs": 5
            },
            "classifier": {
                "in_dim": 64,
                "hidden_layers": [64, 32]
            },
            "classifier_training": {
                "verbose": True,
                "n_epochs": 5
            },
            "seed": 0,
            "visuals": []
        }
        tracking = {}

        tracker = Tracker(**tracking)

        tracker.track(experiment, config)
