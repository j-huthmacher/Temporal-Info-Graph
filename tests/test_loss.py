""" Unit test for the loss

    @author: j-huthmacher
"""
import unittest

import torch
import numpy as np

# pylint: disable=import-error
from model import jensen_shannon_mi
from model.solver import evaluate

class TestEvaluation(unittest.TestCase):
    """ Test class for evaluation function.
    """

    def test_top_k(self):
        """ Simple test to valiadte the top-k evaluation.
        """
        pred = np.array([
            [2,3,5,1,4],
            [1,2,3,4,5],
            [2,4,1,5,3]
        ])
        labels = np.array([2,2,2])

        top1, top5 = evaluate(pred, labels)

        self.assertEqual(top1, 2/3)
        self.assertEqual(top5, 3/3)


class TestLoss(unittest.TestCase):
    """ Test class for TIG loss function.
    """

    def test_js_mi(self):
        """ Simple test for jensen-shannon MI loss.

            Contains also a test for the pos./neg. sampling
        """
        #### Create Artificial Data ####
        enc_global = torch.tensor([
            [1,1,1],
            [2,2,2]
        ])

        enc_local = torch.tensor([
            [
                [3, 3, 3],
                [4, 4, 4],
            ],
            [
                [5, 5, 5],
                [6, 6, 6],
            ]
        ]).permute(0,2,1)

        self.assertTrue(enc_global.shape == (2,3)  and enc_local.shape == (2,3,2))

        loss_fn = jensen_shannon_mi

        loss = loss_fn(enc_global, enc_local)

        #### Check Positive Samples ####
        self.assertTrue(torch.all(torch.eq(loss_fn.pos_samples, torch.tensor([
                                                                        [ 9., 12.],
                                                                        [30., 36.]]))))
        #### Check Negative Samples ####
        self.assertTrue(torch.all(torch.eq(loss_fn.neg_samples, torch.tensor([
                                                                        [ 15., 18.],
                                                                        [18., 24.]]))))
        #### Check Discriminator Matrix ####
        self.assertTrue(torch.all(torch.eq(loss_fn.discr_matr, torch.tensor([
                                                                [ 9., 12., 15., 18.],
                                                                [18., 24., 30., 36.]]))))
                                                                
        #### Positive Expectation ####
        self.assertEqual(round(loss_fn.E_pos.item(), 6), round(0.69311475, 6))

        #### Negative Expectation ####
        # TODO: Recalculate on paper with higher precision
        self.assertEqual(round(loss_fn.E_neg.item(), 2), round(18.05853, 2))

        #### Actual Loss ####
        # TODO: Recalculate on paper with higher precision
        # TODO: E_neg used from python code
        self.assertEqual(round(18.05685 - 0.69311475, 3), round(loss.item(), 3))

    def test_unsupervised_accuracy(self):
        """ 
        """
        #### Create Artificial Data ####
        enc_global = torch.tensor([
            [1,1,1],
            [2,2,2]
        ])

        enc_local = torch.tensor([
            [
                [3, 3, 3],
                [4, 4, 4],
            ],
            [
                [5, 5, 5],
                [6, 6, 6],
            ]
        ]).permute(0,2,1)

        self.assertTrue(enc_global.shape == (2,3)  and enc_local.shape == (2,3,2))

        loss_fn = jensen_shannon_mi

        _ = loss_fn(enc_global, enc_local)

        #### Unsupervised accuracy ####
        # pylint: disable=line-too-long
        labels = torch.block_diag(*[torch.ones(loss_fn.num_nodes) for _ in range(loss_fn.num_graphs)])

        #### Check Discriminator Matrix ####
        self.assertTrue(torch.all(torch.eq(labels, torch.tensor([
                                                        [ 1., 1., 0., 0.],
                                                        [ 0., 0., 1., 1.]]))))

        #### Prediction ####
        # yhat_norm = F.sigmoid(loss_fn.discr_matr)
        yhat_norm = torch.sigmoid(loss_fn.discr_matr)
        # self.assertTrue(torch.all(torch.eq(yhat_norm, torch.tensor([
        #                 [ 0.9999, 1.0000, 1.0000, 1.0000],
        #                 [ 1.0000, 1.0000, 1.0000, 1.0000]]))))
        yhat_norm[yhat_norm > 0.5] = 1
        yhat_norm[yhat_norm <= 0.5] = 0

        acc = (yhat_norm == labels).sum() / torch.numel(labels)

        self.assertEqual(acc, 0.5)
