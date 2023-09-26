import os
import pandas as pd
import numpy as np
import torch
import plot_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = ['path to the dataset']

def get_sampler(sample_dataset, label_name, labels, data_samples, transformed_samples=5):
    class_sample_count = data_samples.groupby("label").size().values*transformed_samples
    sample_dataset.Sample = sample_dataset.Sample.cat.remove_unused_categories()
    sample_labels = np.asarray([[x[1][label_name].cat.codes.iloc[0]] * transformed_samples for x in list(sample_dataset.groupby(['Sample']))]).ravel()
    num_samples = sum(class_sample_count)
    class_weights = [num_samples/class_sample_count[i] for i in range(len(class_sample_count))]
    weights = [class_weights[sample_labels[i]] for i in range(int(num_samples))]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))

    return sampler

def read_tsv(in_file, use_margins=False, set_categories=True, sep='\t'):
    df = pd.read_csv(in_file, sep=sep, header=0, low_memory=False)
    # df = df[(df.Phenotype != 'Potential Artifact') & (df.Phenotype != 'Unclassified') & (df.Phenotype != 'Monocyte') & (df.Phenotype != 'Plasma Cell') & (df.Phenotype != 'Neutrophil')]
        
    if set_categories:
        # Set necessary columns as categorical to later retrieve distinct category codes
        df.Sample = pd.Categorical(df.Sample)
        df.Pathology = pd.Categorical(df.Pathology)
        df.Phenotype = pd.Categorical(df.Phenotype)
        df.Status = pd.Categorical(df.Status)
        df.HLA1_FUNCTIONAL_threeclass = pd.Categorical(df.HLA1_FUNCTIONAL_threeclass)

    return df

def read_dataset(in_file, sub_path, dataset_type = None):
    in_file = path + in_file
    sub_path = path + sub_path
    data = pd.read_csv(sub_path, header=0, low_memory=False)
    df = read_tsv(in_file, set_categories=True)
    df = df[(df['Phenotype'] != "Other")]
    res_data = pd.DataFrame()
    if dataset_type == 'weighted_distance':
        for sample in data.fov:
            lr_id = data[data.fov == sample]["lr_id"]
            reg = data[data.fov == sample]["region"]
            data_to_append = df[df.Sample == sample]
            data_to_append["lr_id"] = len(data_to_append) * [lr_id.values[0]]
            data_to_append["Pathology"] = len(data_to_append) * [reg.values[0]]
            res_data = res_data.append(data_to_append)
    else:
        for sample in data.fov:
            data_to_append = df[df.Sample == sample]
            res_data = res_data.append(data_to_append)
            
    res_data = res_data[(res_data['Phenotype'] != "Unclassified") & (res_data['Phenotype'] != "Potential Artifact")]
    if dataset_type == 'weighted_distance':
        res_data = res_data[["Sample", "X", "Y", "Pathology", "Phenotype", "Status", "lr_id"]]
    else:
        res_data = res_data[["Sample", "X", "Y", "Pathology", "Phenotype", "Status"]]
    res_data.Pathology = pd.Categorical(res_data.Pathology)
    res_data.Phenotype = res_data.Phenotype.cat.remove_unused_categories()
    return res_data, data

def sample_region(dataset, index, label_name, dataset_type):
    '''
    Samples a region (POV) in the dataset.

    Input:
        dataset:    DataFrame consisting of the input data.
        index:      Index of the distinct region to sample.
    '''
    data_at_idx = dataset[dataset.Sample.cat.codes == index]
    label = (int)(data_at_idx[label_name].cat.codes.iloc[0])
    if dataset_type == "weighted_distance":
        features_at_idx = np.vstack(
            (np.array(data_at_idx.X), 
            np.array(data_at_idx.Y), 
            np.array(data_at_idx.Phenotype.cat.codes),
            np.array(data_at_idx.lr_id))
        ).T
    else:
        features_at_idx = np.vstack(
            (np.array(data_at_idx.X), 
            np.array(data_at_idx.Y), 
            np.array(data_at_idx.Phenotype.cat.codes),
)
        ).T

    return torch.FloatTensor(features_at_idx), label

class PathologyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_points, batch_size, train_val_index, label_name, transformed_samples=1, transforms = None, plot_data=False, dataset_type = None):
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
        self.batch_size = batch_size
        self.train_val_index = train_val_index
        self.dataset_type = dataset_type
        
    def __len__(self):
        return len(self.dataset.Sample.cat.categories) * self.transformed_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = int(idx / self.transformed_samples) # Rescale idx into the df shape
        sample, label = sample_region(self.dataset, index=idx, label_name=self.label_name, dataset_type = self.dataset_type)

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
    
