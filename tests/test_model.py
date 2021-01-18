"""

"""
# pylint: disable=not-callable
import unittest

import torch

# pylint: disable=import-error
from model.temporal_info_graph import TemporalInfoGraph

# For reproducibility
torch.manual_seed(0)


class TestModel(unittest.TestCase):
    """ Test class for evaluation function.
    """

    def test_discriminator_reshaping(self):
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
            (2, 6, 9, 7, 2),
            (7, 6, 5, 4, 2)
        ]
        tig = TemporalInfoGraph(architecture=architecture)

        self.assertTrue(torch.equal(
            x, tig.reshape_2d_3d(tig.reshape_3d_2d(x), x.shape)))

    def test_tig_encoder_wo_discr_layer(self):
        """
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
            (2, 6, 9, 7, 2)
        ]
        tig = TemporalInfoGraph(architecture=architecture,
                                A=A, discriminator_layer=False)

        gbl, lcl = tig(X, A)
        self.assertTrue(gbl.shape, (2, 7))  # (num_graphs, emb_dim)
        # (num_graphs, emb_dim, num_nodes)
        self.assertTrue(lcl.shape, (2, 7, 3))

        # torch.set_printoptions(precision=10, sci_mode=False)
        # local_shape = lcl.shape
        # print(lcl.detach())
        # print(lcl.detach().view(-1, -1), lcl.detach().reshape(-1, local_shape[1]).shape)
        # print(gbl.detach())

        #### Check Global Repr ####
        self.assertTrue(torch.allclose(gbl.detach(), torch.tensor([
            [-0.0000061101, 0.4065242708, 0.5743240714, 0.6953476071,
                0.9662133455, 0.3790444136, 0.4473923445]
        ]).type(torch.float32)))

        #### Check Local Repr ####
        self.assertTrue(torch.allclose(lcl.detach(),
                                       torch.tensor([[[0.0000000000, -0.0000122202],
                                                      [0.0000000000, 0.8130485415],
                                                      [1.1487964392, -
                                                          0.0001483500],
                                                      [1.3908934593, -
                                                          0.0001982756],
                                                      [1.9325233698, -
                                                          0.0000967079],
                                                      [0.4877631962, 0.2703256309],
                                                      [-0.0000818149, 0.8948665261]]]).type(torch.float32)))

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
        tig = TemporalInfoGraph(architecture=architecture,
                                A=A, discriminator_layer=False)

        gbl, lcl = tig(X, A)

        self.assertTrue(gbl.shape, (3, 4))  # (num_graphs, emb_dim)
        # (num_graphs, emb_dim, num_nodes)
        self.assertTrue(lcl.shape, (3, 4, 4))

    def test_tig_encoder(self):
        """
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
            (2, 6, 9, 7, 2)
        ]
        tig = TemporalInfoGraph(architecture=architecture, A=A)

        gbl, lcl = tig(X, A)
        self.assertTrue(gbl.shape, (2, 7))  # (num_graphs, emb_dim)
        # (num_graphs, emb_dim, num_nodes)
        self.assertTrue(lcl.shape, (2, 7, 3))

        # torch.set_printoptions(precision=10, sci_mode=False)
        # print(lcl.detach())
        # print(gbl.detach())

        #### Check Global Repr ####
        self.assertTrue(torch.allclose(gbl.detach(), torch.tensor([
            [0.6561244726,  0.1320661604,  0.9772741795,  0.0804385245,
             0.1481097639, -0.2970988154, -0.3897836506]
        ]).type(torch.float32)))

        #### Check Local Repr ####
        self.assertTrue(torch.allclose(lcl.detach(),
                                       torch.tensor([[[-0.3882373571,  0.1420675218],
                                                      [0.8338274956,  0.7872511744],
                                                      [-0.7091836929, -0.3200142682],
                                                      [0.4084744155, -0.1346620917],
                                                      [0.1362719387,  0.2326189876],
                                                      [-0.9435855746, 0.3589177132],
                                                      [-0.7894634604,  0.8177438378]]]).type(torch.float32)))
