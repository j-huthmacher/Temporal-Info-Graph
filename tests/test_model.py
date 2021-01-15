"""
"""
import unittest

import torch

# pylint: disable=import-error
from model.temporal_info_graph import TemporalInfoGraph

# For reproducibility
torch.manual_seed(0)

class TestModel(unittest.TestCase):
    """ Test class for evaluation function.
    """

    def test_tig_encoder(self):
        """
        """
        X = torch.tensor([[
            [
                [1, 2, 1, 1],  # f1, n1 , t1 and t2 and t3 and t4
                [3, 4, 1, 1],  # f1, n2 , t1 and t2 and t3 and t4
                [2, 1, 1, 1]], # f1, n3 , t1 and t2 and t3 and t4
            [
                [2, 2, 1, 1],  # f2, n1 , t1 and t2 and t3 and t4
                [2, 2, 1, 1],  # f2, n2 , t1 and t2 and t3 and t4
                [2, 4, 1, 1]]  # f2, n3 , t1 and t2 and t3 and t4
            ]]).type('torch.FloatTensor').cpu()

        A = torch.tensor([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ]).type('torch.FloatTensor')
        
        # (c_in, c_out, spec_out, out, kernel)
        architecture = [
            (2, 6, 9, 7, 2)
        ]
        tig = TemporalInfoGraph(architecture=architecture, A=A)

        gbl, lcl  = tig(X, A)

        self.assertTrue(gbl.shape, (2, 7))  # (num_graphs, emb_dim)
        self.assertTrue(lcl.shape, (2, 7, 3))  # (num_graphs, emb_dim, num_nodes)
    
    def test_tig_encoder_2layers(self):
        """
        """
        X = torch.rand(3, 2, 4, 10).type('torch.FloatTensor').cpu()

        A = torch.tensor([
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0]
            ]).type('torch.FloatTensor')
        
        # (c_in, c_out, spec_out, out, kernel)
        architecture = [
            (2, 6, 9, 7, 2),
            (7, 6, 5, 4, 2)
        ]
        tig = TemporalInfoGraph(architecture=architecture, A=A)

        gbl, lcl  = tig(X, A)

        self.assertTrue(gbl.shape, (3, 4))  # (num_graphs, emb_dim)
        self.assertTrue(lcl.shape, (3, 4, 4))  # (num_graphs, emb_dim, num_nodes)

