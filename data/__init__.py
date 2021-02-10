"""
"""
# from data.tig_data_set import TIGDataset
from data.data_utils import get_loss, get_normalized_adj

from scipy.linalg import block_diag
import numpy as np 

# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering-in-c-python
A = np.array([
    #0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 0
    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 1
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 4
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 5
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 6 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 7
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 12
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 13
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # 14
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 15
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 17
]) 

KINECT_ADJACENCY = block_diag(A, A)

# https://arxiv.org/pdf/1604.02808.pdf
NTU_ADJACENCY = np.zeros((25, 25))
NTU_ADJACENCY[0, 12] = 1
NTU_ADJACENCY[0, 16] = 1
NTU_ADJACENCY[0, 1] = 1
NTU_ADJACENCY[1, 0] = 1
NTU_ADJACENCY[1, 20] = 1
NTU_ADJACENCY[2, 20] = 1
NTU_ADJACENCY[2, 3] = 1
NTU_ADJACENCY[3, 2] = 1

NTU_ADJACENCY[4, 20] = 1
NTU_ADJACENCY[4, 5] = 1
NTU_ADJACENCY[5, 4] = 1
NTU_ADJACENCY[5, 6] = 1
NTU_ADJACENCY[6, 5] = 1
NTU_ADJACENCY[6, 7] = 1
NTU_ADJACENCY[7, 6] = 1
NTU_ADJACENCY[7, 21] = 1
NTU_ADJACENCY[7, 22] = 1

NTU_ADJACENCY[8, 20] = 1
NTU_ADJACENCY[8, 9] = 1
NTU_ADJACENCY[9, 8] = 1
NTU_ADJACENCY[9, 10] = 1
NTU_ADJACENCY[10, 9] = 1
NTU_ADJACENCY[10, 11] = 1
NTU_ADJACENCY[11, 10] = 1
NTU_ADJACENCY[11, 23] = 1
NTU_ADJACENCY[11, 24] = 1

NTU_ADJACENCY[12, 0] = 1
NTU_ADJACENCY[12, 13] = 1
NTU_ADJACENCY[13, 12] = 1
NTU_ADJACENCY[13, 14] = 1
NTU_ADJACENCY[14, 13] = 1
NTU_ADJACENCY[14, 15] = 1
NTU_ADJACENCY[15, 14] = 1

NTU_ADJACENCY[16, 0] = 1
NTU_ADJACENCY[16, 17] = 1
NTU_ADJACENCY[17, 16] = 1
NTU_ADJACENCY[17, 18] = 1
NTU_ADJACENCY[18, 17] = 1
NTU_ADJACENCY[18, 19] = 1
NTU_ADJACENCY[19, 18] = 1

NTU_ADJACENCY[20, 1] = 1
NTU_ADJACENCY[20, 8] = 1
NTU_ADJACENCY[20, 4] = 1
NTU_ADJACENCY[20, 2] = 1

NTU_ADJACENCY[21, 7] = 1
NTU_ADJACENCY[22, 7] = 1

NTU_ADJACENCY[23, 11] = 1
NTU_ADJACENCY[24, 11] = 1


