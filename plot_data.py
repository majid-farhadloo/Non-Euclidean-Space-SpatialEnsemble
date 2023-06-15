from itertools import permutations

import dataset, spatial_utils

import pandas as pd, numpy as np
import torch

import os

import matplotlib.pyplot as plt, matplotlib.pylab as pylab

from sklearn.manifold import TSNE
from tqdm import tqdm

def get_phenotype_cmap():
    '''
    Maps cell types to distinctive colors for plotting.

    Output:
        phenotype_dict: Dict of phenotype:color mappings.
    '''

    phenotype_dict = {
        'Tumor Cell': 'black',
        'Unclassified': 'grey',
        'Potential Artifact': 'lightgrey',
        'Macrophage': 'magenta',
        'B Cell': 'red',
        'Monocyte': 'orange',
        'Helper T Cell': 'greenyellow',
        'Regulatory T Cell': 'springgreen',
        'Cytotoxic T Cell': 'cyan',
        'Vasculature': 'royalblue',
        'Neutrophil': 'blueviolet',
        'Plasma Cell': 'lightpink',
    }

    return phenotype_dict

def get_phenotype_cmap2():
    '''palette = {"B Cells":"#0000FF","Cytotoxic T Cell":"tab:cyan", "Helper T Cells":"#000000", 
    "Lymphocyte Other":"#1D8348", "Macrophage":"#34FF70", "Monocyte": "#3CAB97", "Myeloid Cell Other":"#9999FF", 
    "NK Cells":"#ffff00","Neutrophils":"#FFB8CE","Plasma Cells":"#FF1493", "Regulatory T Cell": "#884EA0",
    "Tumor":"#ff0000","Unclassified":"#CCCCCC", "Vasculature":"#D4AB84", "Neutrophil" : "#F665E9"}'''

    phenotype_dict = {
        "B Cell":"#0000FF",
        "Cytotoxic T Cell":"tab:cyan", 
        "Helper T Cell":"#000000", 
        "Lymphocyte Other":"#1D8348", 
        "Macrophage":"#34FF70", 
        "Monocyte": "#3CAB97", 
        "Myeloid Cell Other":"#9999FF", 
        "NK Cells":"#ffff00",
        "Neutrophil":"#FFB8CE",
        "Plasma Cell":"#FF1493", 
        "Regulatory T Cell": "#884EA0",
        "Tumor Cell":"#ff0000",
        "Unclassified":"#CCCCCC", 
        "Vasculature":"#D4AB84", 
        #"Neutrophil" : "#F665E9"
    }

    return phenotype_dict

def plot_sample(sample, num_points=None):
    '''
    Plots the sample region along with the cmap legend.

    Input:
        sample: DataFrame containing the sample region to plot.
    '''

    # Load the phenotype cmap and retrieve sample info for the plot title
    phenotype_cmap = get_phenotype_cmap()
    sample_name = sample.iloc[0].Sample
    pathology = sample.iloc[0].Pathology

    if num_points:
        sample = sample[:num_points]

    fig = pylab.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    for phenotype in phenotype_cmap:
        # If this is a tumor cell, apply a marker corresponding to its HLA1 type
        if phenotype == 'Tumor Cell':
            for (hla1_type, marker) in [('Negative', 'v'), ('Moderate', 'o'), ('High', '^')]:
                phenotype_rows = sample[(sample.Phenotype == phenotype) & (sample.HLA1_FUNCTIONAL_threeclass == hla1_type)]
                ax.scatter(x=phenotype_rows.X, y=phenotype_rows.invertY, s=9, c=phenotype_cmap[phenotype], label=phenotype+': HLA1 '+hla1_type, marker=marker)
        else:
            phenotype_rows = sample[sample.Phenotype == phenotype]
            ax.scatter(x=phenotype_rows.X, y=phenotype_rows.invertY, s=4, c=phenotype_cmap[phenotype], label=phenotype)
    
    plt.title('Sample: %s, %s' % (sample_name, pathology))
    fig.show()

    figlegend = pylab.figure(figsize=(3,4))
    figlegend.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])
    figlegend.show()

def plot_data(xs, ts, df, n_dist=None, plot_legend=False):
    phenotype_cmap = get_phenotype_cmap2()
    coords = np.array(xs[:,:2])
    phenotypes = np.array(xs[:,2])
    if ts==0: label = 'Intact'
    if ts==1: label = 'Mixed'
    #if ts[b].eq(torch.tensor([0, 0, 1])).all(): label = 'Tumor'

    fig = pylab.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    for phenotype in phenotype_cmap:
        if not [x for x in df.Phenotype.cat.categories if x == phenotype]:
            continue
        phenotype_coords = coords[phenotypes == np.argwhere(df.Phenotype.cat.categories == phenotype)[0]]
        b_cell = np.argwhere(df.Phenotype.cat.categories == 'B Cell')[0]
        tumor_cell = np.argwhere(df.Phenotype.cat.categories == 'Tumor Cell')[0]
        helpert_cell = np.argwhere(df.Phenotype.cat.categories == 'Helper T Cell')[0]
        if n_dist:
            #for neighbor_phenotype in tqdm(np.argwhere(df.Phenotype.cat.categories != phenotype)):
            if phenotype == 'B Cell':
                neighbor_coords = coords[phenotypes == tumor_cell]
                dist_matrix = spatial_utils.build_distance_matrix(phenotype_coords, neighbor_coords, n_dist)
                
                center_coords = phenotype_coords[np.argwhere(dist_matrix > 0)[:,0]]
                c_x, c_y = [x[0] for x in center_coords], [y[1] for y in center_coords]
                neighbor_coords = neighbor_coords[np.argwhere(dist_matrix > 0)[:,1]]
                n_x, n_y = [x[0] for x in neighbor_coords], [y[1] for y in neighbor_coords]

                plt.plot([c_x, n_x], [c_y, n_y], c='black', linewidth=0.4)

        ax.scatter(x=phenotype_coords[:,0], y=phenotype_coords[:,1], s=4, c=phenotype_cmap[phenotype], label=phenotype)
    
    plt.title(f'Sample: {df.Sample.iloc[0]}, Label: {label}')
    fig.show()

    if plot_legend:
        figlegend = pylab.figure(figsize=(3,4))
        figlegend.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])
        figlegend.show()

if __name__ == '__main__':
    in_file='datasets/BestClassification_July2021_14Samples.tsv'

    df = dataset.read_tsv(in_file)
    cats = df.Sample.cat.categories
    print('Number of categories: %i' % (len(cats)))

    '''List of cells and HLA1s to process'''
    cells = [
        ('Tumor Cell', 'Negative'),
        ('Unclassified', 'NA'),
        ('Macrophage', 'NA'),
        ('B Cell', 'NA'),
        ('Helper T Cell', 'NA'),
        ('Regulatory T Cell', 'NA'),
        ('Cytotoxic T Cell', 'NA'),
        ('Vasculature', 'NA'),
        ('Neutrophil', 'NA'),
        ('Plasma Cell', 'NA'),
    ]

    region = ['Normal', 'Tumor', 'Interface']

    #sample = dataset.sample_region(df, index=4)
    #plot_sample(sample)

    '''Find all unique n_perms in the set of cells'''
    n_perms = 2
    perms = [x for x in permutations(cells,n_perms) if x[:][1][0] != 'Tumor Cell']# and x[:][2][0] != 'Tumor Cell']
    print(f'Number of permutations: {len(perms)}')

    '''Arrange the output dict, which will become a CSV'''
    data = {}
    data['Region'] = []
    for n in range(n_perms):
        data['Phenotype'+str(n)]=[]
    data['Instances'], data['Counts'], data['PRs'], data['Label'] = [], [], [], []

    '''Iterate over each sample region in the dataset'''
    for cat in tqdm(cats):
        sample, label = dataset.sample_region_df(df, cat=cat)
        perms = permutations(cells,n_perms)
        
        '''For each n_perm, where the second and third rows are not Tumor Cell (to remove tumor cell -> tumor cell interactions)'''
        for perm in [x for x in perms if x[:][1][0] != 'Tumor Cell']:# and x[:][2][0] != 'Tumor Cell']:
            data['Region'].append(cat)
            data['Label'].append(label)

            '''Load the phenotype names into the dict'''
            sample_ptypes = []
            for n, p in enumerate(perm):
                s = p[0]
                '''if s=='Tumor Cell':
                    s+=': '+p[1]'''
                data['Phenotype'+str(n)].append(s)
                sample_ptypes.append(sample[sample.Phenotype == p[0]])
            #if sample.iloc[0].Pathology != 'Normal': continue

            # sample_name = cat
            pathology = sample.iloc[0].Pathology
            # print('Sample: %s, %s Region' % (sample_name, pathology))
            # print('The sample size is %i' % len(sample))
            target_cells = cells[3:]

            # print('Number of %s: %i' % (ptype1, len(sample_ptype1)))
            # print('Number of %s: %i' % (ptype2, len(sample_ptype2)))

            instances = spatial_utils.calculate_instances(sample_ptypes[0], sample_ptypes[1:], d=50)
            count = len(sample_ptypes[0])
            # print('Participation Ratio: %0.3f' % (instances / count))

            data['Instances'].append(instances)
            data['Counts'].append(count)
        
            if data['Counts'][-1]>0:
                data['PRs'].append(data['Instances'][-1]/data['Counts'][-1])
            else:
                data['PRs'].append(0)
        
    '''Output result to CSV'''
    df = pd.DataFrame(data) 
    
    # saving the dataframe 
    df.to_csv('all_doubles.csv')
    print()

def attn_to_csv(attention, a, Wh, stack, out_file):
    if not os.path.exists('outputs/attn/' + out_file):
        attn_mat = torch.einsum('bnkf, bnkf -> bnk', attention, Wh)

        l = [[attn_mat[(stack[0,:,:,:] == i) & (stack[1,:,:,:] == j)].mean().item() for i in range(int(stack.max().item()+1))] for j in range(int(stack.max().item()+1))]            
        
        csv = {}
        for i, n in enumerate(l):
            n.append(np.nan)
            csv[str(i)] = n
        l_end = list(np.empty((13)) * np.nan)
        l_end[-1] = attn_mat[(stack[0,:,:,:] == -1)].mean().item()
        csv[str(stack.max().item()+1)] = l_end

        df = pd.DataFrame(csv)
        df.to_csv('outputs/attn/' + out_file)