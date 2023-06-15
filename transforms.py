import torch
import math

import numpy as np

def get_rotated_data(raw_data, n_rotations):
    '''
    Computes n-way rotation on the data.
    '''
    x_avg = torch.mean(raw_data[:, 0].float())
    y_avg = torch.mean(raw_data[:, 1].float())

    x = raw_data[:, 0] - x_avg
    y = raw_data[:, 1] - y_avg

    ss = torch.sin(torch.arange(0, 2 * math.pi - 2 * math.pi / n_rotations + 1e-4, 2 * math.pi / n_rotations))
    cs = torch.cos(torch.arange(0, 2 * math.pi - 2 * math.pi / n_rotations + 1e-4, 2 * math.pi / n_rotations))

    x_new = torch.outer(x, cs) - torch.outer(y, ss) + x_avg
    y_new = torch.outer(x, ss) + torch.outer(y, cs) + y_avg

    r_idx = np.random.randint(0, n_rotations)
    res = raw_data
    res[:, 0] = x_new[:, r_idx] - torch.min(x_new[:, r_idx])
    res[:, 1] = y_new[:, r_idx] - torch.min(y_new[:, r_idx])
    return res

class RotateData(object):
    """
    Rotates the data randomly.
    """

    def __init__(self, n_rotations=32):
        self.n_rotations = n_rotations

    def __call__(self, sample):
        # sample_data = torch.FloatTensor(sample)
        sample_data = sample
        rotated_data = get_rotated_data(sample_data, self.n_rotations)

        return rotated_data

def get_partial_data(raw_data, n_partitions=2, overlap_ratio=0.8):
    '''
    Gets n_partitions subsets of the data.
    '''
    x_min = torch.min(raw_data[:, 0]).item()
    x_max = torch.max(raw_data[:, 0]).item() + 1
    y_min = torch.min(raw_data[:, 1]).item()
    y_max = torch.max(raw_data[:, 1]).item() + 1

    x_idx = np.random.randint(0, n_partitions) #Sampled on a grid of size n_partitions by n_partitions
    y_idx = np.random.randint(0, n_partitions) #Sampled on a grid of size n_partitions by n_partitions
    
    x_dist = ((x_max - x_min) / n_partitions)
    y_dist = ((y_max - y_min) / n_partitions)

    lo_x = x_min + (x_dist * x_idx * overlap_ratio)
    hi_x = lo_x + (x_dist * (1+overlap_ratio))

    lo_y = y_min + (y_dist * y_idx * overlap_ratio)
    hi_y = lo_y + (y_dist * (1+overlap_ratio))

    partial_data = raw_data[torch.all(
        torch.stack((
            raw_data[:, 0] >= lo_x,
            raw_data[:, 0] <= hi_x,
            raw_data[:, 1] >= lo_y,
            raw_data[:, 1] <= hi_y
        ), dim=1), dim=1
    )]

    return partial_data

class PartitionData(object):
    """
    Takes a random partition of the data.
    """

    def __init__(self, n_partitions=2, n_rotations=32):
        self.n_partitions = n_partitions
        self.n_rotations = n_rotations

    def __call__(self, sample):
        sample_data = torch.FloatTensor(sample)
        rotated_data = get_partial_data(sample_data, self.n_partitions)
        rotated_data = get_rotated_data(sample_data, self.n_rotations)

        return rotated_data
