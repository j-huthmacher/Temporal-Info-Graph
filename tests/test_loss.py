""" Unit test for the differtent loss functions.
"""
# pylint: disable=wrong-import-position, import-error, not-callable
import os
import sys

import torch
# import numpy as np

# Workaround to import relative parent packages.
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

# pylint: disable=import-error
from model import jensen_shannon_mi, bce_loss


def test_bce():
    """ Unit test to verify the BCE Loss implementation.
    """
    enc_global = torch.tensor([
        [1, 1],
        [2, 2],
        [9, 9]
    ])

    enc_local = torch.tensor([
        [
            [3, 3],
            [4, 4],
            [7, 7],
            [8, 8],
        ],
        [
            [5, 5],
            [6, 6],
            [10, 10],
            [11, 11],
        ],
        [
            [12, 12],
            [13, 13],
            [14, 14],
            [15, 15],
        ]
    ]).permute(0, 2, 1)

    assert (tuple(enc_global.shape) == (3, 2) and enc_local.shape == (3, 2, 4))

    loss_fn = bce_loss

    _ = bce_loss(enc_global, enc_local)

    #### Check Discriminator Matrix ####
    check_tensor = torch.tensor([
        [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]
    ])
    assert torch.all(torch.eq(loss_fn.mask, check_tensor))

    yhat_norm = torch.sigmoid(loss_fn.discr_matr)
    yhat_norm[yhat_norm > 0.5] = 1
    yhat_norm[yhat_norm <= 0.5] = 0

    loss_acc = (yhat_norm == loss_fn.mask).sum() / torch.numel(loss_fn.mask)

    assert loss_acc == 1 / 3


def test_js_mi():
    """ Unit test to verify the Jensen-Shannon MI loss implementation.

        Contains also a test for the pos./neg. sampling
    """
    #### Create Artificial Data ####
    # 3 graphs with emb_dim of 3
    enc_global = torch.tensor([
        [1, 1],
        [2, 2],
        [9, 9]
    ])

    # Each graph has 4 nodes
    enc_local = torch.tensor([
        [
            [3, 3, 3],
            [4, 4, 4],
            [7, 7, 7],
            [8, 8, 8],
        ],
        [
            [5, 5, 5],
            [6, 6, 6],
            [10, 10, 10],
            [11, 11, 11],
        ]
    ]).permute(2, 0, 1)

    assert enc_global.shape == (3, 2) and enc_local.shape == (3, 2, 4)

    loss_fn = jensen_shannon_mi

    _ = loss_fn(enc_global, enc_local)

    #### Check Positive Samples ####
    check_tensor = torch.tensor([
        [8., 10., 17., 19.],
        [16., 20., 34., 38.],
        [72., 90., 153., 171.]
    ])
    assert torch.all(torch.eq(loss_fn.pos_samples, check_tensor))

    #### Check Negative Samples ####
    check_tensor = torch.tensor([
        [8., 10., 17., 19., 8., 10., 17., 19.],
        [16., 20., 34., 38., 16., 20., 34., 38.],
        [72., 90., 153., 171., 72., 90., 153., 171.]
    ])
    assert torch.all(torch.eq(loss_fn.neg_samples, check_tensor))

    #### Check Discriminator Matrix ####
    check_tensor = torch.tensor([
        [8., 10., 17., 19., 8., 10., 17., 19., 8., 10., 17., 19.],
        [16., 20., 34., 38., 16., 20., 34., 38., 16., 20., 34., 38.],
        [72., 90., 153., 171., 72., 90., 153., 171., 72., 90., 153., 171.]
    ])
    assert torch.all(torch.eq(loss_fn.discr_matr, check_tensor))


def test_unsupervised_accuracy():
    """ Unit test to further verify the Jensen-Shannon MI loss implementation, by the
        checking the unsupervised accuracy, i.e. the accuracy to predict neg. and pos.
        samples correctly.
    """
    #### Create Artificial Data ####
    enc_global = torch.tensor([
        [1, 1, 1],
        [2, 2, 2]
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
    ]).permute(0, 2, 1)

    assert enc_global.shape == (2, 3) and enc_local.shape == (2, 3, 2)

    loss_fn = jensen_shannon_mi

    _ = loss_fn(enc_global, enc_local)

    #### Unsupervised accuracy ####
    labels = torch.block_diag(*[torch.ones(loss_fn.num_nodes)
                                for _ in range(loss_fn.num_graphs)])

    #### Check Discriminator Matrix ####
    assert torch.all(torch.eq(labels, torch.tensor([
        [1., 1., 0., 0.],
        [0., 0., 1., 1.]])))

    #### Prediction ####
    # yhat_norm = F.sigmoid(loss_fn.discr_matr)
    yhat_norm = torch.sigmoid(loss_fn.discr_matr)
    # self.assertTrue(torch.all(torch.eq(yhat_norm, torch.tensor([
    #                 [ 0.9999, 1.0000, 1.0000, 1.0000],
    #                 [ 1.0000, 1.0000, 1.0000, 1.0000]]))))
    yhat_norm[yhat_norm > 0.5] = 1
    yhat_norm[yhat_norm <= 0.5] = 0

    acc = (yhat_norm == labels).sum() / torch.numel(labels)

    assert acc == 0.5
