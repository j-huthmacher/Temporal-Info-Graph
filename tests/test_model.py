""" PyTest to test the model.
"""
# pylint: disable=wrong-import-position, import-error
import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import torch

from model.temporal_info_graph import TemporalInfoGraph, TemporalConvolution, SpectralConvolution

# For reproducibility
torch.manual_seed(0)


def test_tig_dimension():
    """ Test the dimensions of the TIG encoder.
    """
    X = torch.tensor([[
        [
            [1, 2, 1, 1],  # f1, n1 , t1 and t2 and t3 and t4
            [3, 4, 1, 1],  # f1, n2 , t1 and t2 and t3 and t4
            [2, 1, 1, 1]],  # f1, n3 , t1 and t2 and t3 and t4
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
        (2, 6, 12, 12, 2)
    ]
    tig = TemporalInfoGraph(architecture=architecture,
                            A=A, discriminator_layer=False)
    gbl, lcl = tig(X)
    assert tuple(gbl.shape) == (1, 12)  # (num_graphs, emb_dim)
    assert tuple(lcl.shape) == (1, 12, 3)  # (num_graphs, emb_dim, num_nodes)


def test_discriminator_reshaping():
    """ Make sure that the reshaping operation for the discriminator FF is valid
    """
    x = torch.tensor([
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [7, 8, 9],
            [10, 11, 12],
        ]
    ])

    # Not used
    architecture = [
        (2, 6, 12, 12, 2),
        (12, 24, 24, 24, 2)
    ]
    tig = TemporalInfoGraph(architecture=architecture)

    assert torch.equal(x, tig.reshape_2d_3d(tig.reshape_3d_2d(x), x.shape))


def test_tig_encoder_wo_discr_layer():
    """ Test the dimensions withoit discriminator layer.
    """
    X = torch.tensor([[
        [
            [1, 2, 1, 1],  # f1, n1 , t1 and t2 and t3 and t4
            [3, 4, 1, 1],  # f1, n2 , t1 and t2 and t3 and t4
            [2, 1, 1, 1]],  # f1, n3 , t1 and t2 and t3 and t4
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
        (2, 6, 12, 12, 2)
    ]
    tig = TemporalInfoGraph(architecture=architecture,
                            A=A, discriminator_layer=False)

    gbl, lcl = tig(X)
    assert tuple(gbl.shape) == (1, 12)  # (num_graphs, emb_dim)
    assert tuple(lcl.shape) == (1, 12, 3)  # (num_graphs, emb_dim, num_nodes)

    # torch.set_printoptions(precision=10, sci_mode=False)

    #### Check Global Repr ####
    # result = torch.tensor([
    #     [0.0000000000, 0.2619505823, 0.5360082984, 0.7373406291, 0.9563336372,
    #      0.1766242385, 0.4704684317]
    # ])
    # assert torch.allclose(gbl.detach(), result)

    # #### Check Local Repr ####
    # result = torch.tensor([[[0.0000000000, 0.0000000000, 0.0000000000],
    #                         [0.0000000000, 0.7858517766, 0.0000000000],
    #                         [0.0000000000, 0.5193455219, 1.0886794329],
    #                         [0.0000000000, 1.7144789696, 0.4975428283],
    #                         [0.7925892472, 1.2838224173, 0.7925892472],
    #                         [0.0000000000, 0.0000000000, 0.5298727155],
    #                         [0.0000000000, 1.4114053249, 0.0000000000]]])
    # assert torch.allclose(lcl.detach(), result)


def test_tig_encoder_2layers():
    """ Test the output dimensions for a 2 layer encoder.
    """
    # (batch_size, features, nodes, time)
    X = torch.rand(3, 2, 4, 10).type('torch.FloatTensor').cpu()

    A = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0]
    ]).type('torch.FloatTensor')

    # (c_in, c_out, spec_out, out, kernel)
    architecture = [
        (2, 6, 12, 12, 2),
        (12, 24, 24, 24, 2)
    ]
    tig = TemporalInfoGraph(architecture=architecture,
                            A=A, discriminator_layer=False)

    gbl, lcl = tig(X)

    assert gbl.shape == (3, 24)  # (num_graphs, emb_dim)
    assert lcl.shape == (3, 24, 4)  # (num_graphs, emb_dim, num_nodes)


def test_tig_encoder():
    """ Test the encoder dimensions with an one layer architecture.
    """
    X = torch.tensor([[
        [
            [1, 2, 1, 1],  # f1, n1 , t1 and t2 and t3 and t4
            [3, 4, 1, 1],  # f1, n2 , t1 and t2 and t3 and t4
            [2, 1, 1, 1]],  # f1, n3 , t1 and t2 and t3 and t4
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
        (2, 6, 12, 12, 2)
    ]
    tig = TemporalInfoGraph(architecture=architecture, A=A)

    gbl, lcl = tig(X)
    assert tuple(gbl.shape) == (1, 12)  # (num_graphs, emb_dim)
    assert tuple(lcl.shape) == (1, 12, 3)  # (num_graphs, emb_dim, num_nodes)

    # torch.set_printoptions(precision=10, sci_mode=False)

    #### Check Global Repr ####
    # resutl = torch.tensor([
    #     [0.5572497249, 0.1188270152, 0.9377672672, 0.0637857318,
    #      0.1603138745, -0.4778861105, -0.2595149279]
    # ])
    # assert torch.allclose(gbl.detach(), resutl)

    # #### Check Local Repr ####
    # result = torch.tensor([[[0.5909340382, -0.2401582003, -0.4284988344],
    #                         [1.1895797253, 0.9177638888, 0.8900489807],
    #                         [-0.3513569832, -0.6072705984, -0.6237034798],
    #                         [-0.0387212709, -0.0507152379, 0.1812662780],
    #                         [-0.1434196979, 0.6819714308, 0.4132525027],
    #                         [0.5097138882, -0.6692208052, -0.2487553656],
    #                         [0.3655797243, -0.0922359824, 0.2696639299]]])
    # assert torch.allclose(lcl.detach(), result)


def test_tig_res_e_weights():
    """ Test the encoder with residual layer and edge weights.
    """
    X = torch.tensor([[
        [
            [1, 2, 1, 1],  # f1, n1 , t1 and t2 and t3 and t4
            [3, 4, 1, 1],  # f1, n2 , t1 and t2 and t3 and t4
            [2, 1, 1, 1]],  # f1, n3 , t1 and t2 and t3 and t4
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
        (2, 6, 12, 12, 2)
    ]
    tig = TemporalInfoGraph(architecture=architecture,
                            A=A, residual=True, edge_weights=True)

    gbl, lcl = tig(X)
    assert (tuple(X.shape) == (1, 2, 3, 4))  # (batch, features, nodes, time)
    assert (tuple(gbl.shape) == (1, 12))  # (num_graphs, emb_dim)
    assert (tuple(lcl.shape) == (1, 12, 3))  # (num_graphs, emb_dim, num_nodes)


def test_temp_cov_all_features():
    """ Test the calculation of the temporal convolution (over all features).
    """
    # Predefined example tensor for verification
    X = torch.tensor([[
        [[1, 2, 1, 1],   # f1, n1 , t1 and t2 and t3 and t4
         [3, 4, 1, 1],   # f1, n2 , t1 and t2 and t3 and t4
         [2, 1, 1, 1]],  # f1, n3 , t1 and t2 and t3 and t4
        [[2, 2, 1, 1],   # f2, n1 , t1 and t2 and t3 and t4
         [2, 2, 1, 1],   # f2, n2 , t1 and t2 and t3 and t4
         [2, 4, 1, 1]]   # f2, n3 , t1 and t2 and t3 and t4
    ]]).type('torch.FloatTensor').cpu()

    assert X.shape == (1, 2, 3, 4)

    c_in = 2
    c_out = 1
    weights = torch.ones((c_out, 2, 1, 2)).type('torch.FloatTensor')

    tempConv = TemporalConvolution(
        c_in=2, c_out=1, weights=weights, debug=True, per_feature=False).cpu()

    yhat = tempConv(X)

    result = torch.tensor([[[[7., 6., 4.],
                             [11., 8., 4.],
                             [9., 7., 4.]]]])
    assert torch.allclose(yhat, result)


def test_temp_cov_per_features():
    """ Test the calculation of the temporal convolution (per feature).
    """
    # Predefined example tensor for verification
    X = torch.tensor([[
        [[1, 2, 1, 1],   # f1, n1 , t1 and t2 and t3 and t4
         [3, 4, 1, 1],   # f1, n2 , t1 and t2 and t3 and t4
         [2, 1, 1, 1]],  # f1, n3 , t1 and t2 and t3 and t4
        [[2, 2, 1, 1],   # f2, n1 , t1 and t2 and t3 and t4
         [2, 2, 1, 1],   # f2, n2 , t1 and t2 and t3 and t4
         [2, 4, 1, 1]]   # f2, n3 , t1 and t2 and t3 and t4
    ]]).type('torch.FloatTensor').cpu()

    assert X.shape == (1, 2, 3, 4)

    c_in = 2
    c_out = c_in
    weights = torch.ones((c_out, 1, 1, 2)).type('torch.FloatTensor')

    tempConv = TemporalConvolution(
        c_in, c_out, weights=weights, debug=True, per_feature=True).cpu()

    yhat = tempConv(X)

    result = torch.tensor([[
        [[3., 3., 2.],
         [7., 5., 2.],
         [3., 2., 2.]],
        [[4., 3., 2.],
         [4., 3., 2.],
         [6., 5., 2.]]]])
    assert torch.allclose(yhat, result)


def test_spec_cov():
    """ Test the calculation of the spectral convolution.
    """
    # Predefined example tensor for verification
    X = torch.tensor([[
        [[1, 2, 1, 1],   # f1, n1 , t1 and t2 and t3 and t4
         [3, 4, 1, 1],   # f1, n2 , t1 and t2 and t3 and t4
         [2, 1, 1, 1]],  # f1, n3 , t1 and t2 and t3 and t4
        [[2, 2, 1, 1],   # f2, n1 , t1 and t2 and t3 and t4
         [2, 2, 1, 1],   # f2, n2 , t1 and t2 and t3 and t4
         [2, 4, 1, 1]]   # f2, n3 , t1 and t2 and t3 and t4
    ]]).type('torch.FloatTensor').cpu()

    assert X.shape == (1, 2, 3, 4)

    A = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]).type('torch.FloatTensor').cpu()

    assert A.shape == (X.shape[2], X.shape[2])

    # We need a matrix to have proper dimensions for the permutation
    weights = torch.tensor([1., 1.]).type('torch.FloatTensor').unsqueeze(dim=1)

    specConv = SpectralConvolution(c_in=2, c_out=1, weights=weights,
                                   debug=True, A=A).cpu()

    yhat = specConv(X)

    result = torch.tensor([[[[8., 10., 4., 4.],
                             [12., 15., 6., 6.],
                             [9., 11., 4., 4.]]]])
    assert torch.allclose(yhat, result)


def test_spec_conv_dim():
    """ Test the dimensions of the spectral convolution.
    """
    # Predefined example tensor for verification
    X = torch.tensor([[
        [[1, 2, 1, 1],   # f1, n1 , t1 and t2 and t3 and t4
         [3, 4, 1, 1],   # f1, n2 , t1 and t2 and t3 and t4
         [2, 1, 1, 1]],  # f1, n3 , t1 and t2 and t3 and t4
        [[2, 2, 1, 1],   # f2, n1 , t1 and t2 and t3 and t4
         [2, 2, 1, 1],   # f2, n2 , t1 and t2 and t3 and t4
         [2, 4, 1, 1]]   # f2, n3 , t1 and t2 and t3 and t4
    ]]).type('torch.FloatTensor').cpu()

    assert X.shape == (1, 2, 3, 4)

    A = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]).type('torch.FloatTensor').cpu()

    assert A.shape == (X.shape[2], X.shape[2])

    specConv = SpectralConvolution(c_in=2, c_out=16, A=A).cpu()

    yhat = specConv(X)

    assert yhat.shape == (1, 16, 3, 4)
