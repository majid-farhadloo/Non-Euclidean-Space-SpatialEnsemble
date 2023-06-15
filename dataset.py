import os

import pandas as pd
import numpy as np
import torch
import sys
import plot_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = '[SVANN_Directory]'

def get_sampler(sample_dataset, label_name, labels, transformed_samples=5):
    class_sample_count = [len([x[0] for x in list(sample_dataset.groupby([label_name, 'Sample'])) if x[0][0] == labels[i]]) * transformed_samples for i in range(len(labels))]
    sample_labels = np.asarray([[x[1][label_name].cat.codes.iloc[0]] * transformed_samples for x in list(sample_dataset.groupby(['Sample']))]).ravel()

    num_samples = sum(class_sample_count)

    class_weights = [num_samples/class_sample_count[i] for i in range(len(class_sample_count))]
    weights = [class_weights[sample_labels[i]] for i in range(int(num_samples))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))

    return sampler

def read_tsv(in_file, use_margins=False, set_categories=True, sep='\t'):
    df = pd.read_csv(in_file, sep=sep, header=0, low_memory=False)
    
    if set_categories:
        # Set necessary columns as categorical to later retrieve distinct category codes
        df.Sample = pd.Categorical(df.Sample)
        df.place_type = pd.Categorical(df.place_type)
        df.point_categories = pd.Categorical(df.point_categories)
        df.class_label = pd.Categorical(df.class_label)
    return df

# def read_tumor_core_samples(in_file, tumor_core_file):

def read_dataset(in_file, sub_path, sub_data = None, zonal= False, dataset = None):
    in_file = path + in_file
    if zonal:
        sub_path = path + sub_path
        data = pd.read_csv(sub_path, sep=',', header=0, low_memory=False)
        df = read_tsv(in_file, set_categories=False)
    else:
        data = pd.DataFrame(sub_data)
        df = read_tsv(in_file, set_categories=False)  
    
    res_data = pd.DataFrame()
    for sample in data.Sample:
        res_data = res_data.append(df[df.Sample == sample])

    df = res_data
    # Set necessary columns as categorical to later retrieve distinct category codes
    df.Sample = pd.Categorical(df.Sample)
    df.point_categories = pd.Categorical(df.point_categories)
    df.class_label = pd.Categorical(df.class_label)
    df.place_type = pd.Categorical(df.place_type)

    return df

def sample_region(dataset, index, label_name):
    '''
    Samples a region (FOV) in the dataset.

    Input:
        dataset:    DataFrame consisting of the input data.
        index:      Index of the distinct region to sample.
    '''
    data_at_idx = dataset[dataset.Sample.cat.codes == index]
    label = (int)(data_at_idx[label_name].cat.codes.iloc[0])

    features_at_idx = np.vstack(
        (np.array(data_at_idx.X), 
        np.array(data_at_idx.Y), 
        np.array(data_at_idx.point_categories.cat.codes))
    ).T

    return torch.FloatTensor(features_at_idx), label

class place_typeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_points, class_label, batch_size, num_samples, epoch, train_val_index, label_name, num_neighbors = 5, 
                 radius = 75, transformed_samples=1, dataset_name= None, transforms = None, plot_data=False):
    # def __init__(self, dataset, num_points=None, transforms=None, transformed_samples=1, label_name='Margin', plot_data=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = dataset
        self.transforms = transforms
        self.transformed_samples = transformed_samples
        self.num_points = num_points
        self.label_name = label_name
        self.plot_data = plot_data
        self.class_label = class_label
        self.batch_size = batch_size
        self.epoch = epoch
        self.dataset_name = dataset_name
        self.train_val_index = train_val_index
        self.num_neighbors = num_neighbors
        self.radius = radius
        
    def __len__(self):
        return len(self.dataset.Sample.cat.categories) * self.transformed_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = int(idx / self.transformed_samples) # Rescale idx into the df shape
        sample, label = sample_region(self.dataset, index=idx, label_name=self.label_name)

        ## NOTE: For sanity checking
        if self.plot_data:
            data_at_idx = self.dataset[self.dataset.Sample.cat.codes == idx]
            plot_data.plot_data(sample, label, data_at_idx, n_dist=50)
            print()

        if self.transforms:
            sample = self.transforms(sample)
        
        if self.num_points != None and self.num_points < len(sample):
            indices = np.random.choice(len(sample), self.num_points, replace=False)
            sample = sample[indices] # Random sampling

        return sample, label
    
def collate_fn_pad(batch):
    '''
    Zero-pads batch of variable length, so that DataLoader doesn't explode
    '''
    seq = [a_tuple[0] for a_tuple in batch]
    labels = torch.tensor([a_tuple[1] for a_tuple in batch])
    lengths = torch.tensor([ t.shape[0] for t in seq ]).to(device)
    seq = torch.nn.utils.rnn.pad_sequence(seq, padding_value=-1)

    return seq, labels, lengths
