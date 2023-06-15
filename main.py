from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from base_model import PointNet, DGCNN, FCNN
from SAMCNet import SpatialDGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import pandas as pd
import time as time
from datetime import datetime
import dataset, transforms
import dataset
import torchvision.transforms
import pickle
import warnings
from torch.profiler import profile, record_function, ProfilerActivity 
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _init_():
    add =  './checkpoints/' + args.dataset + "/" + args.exp_name 
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(add):
        os.makedirs(add)
    if not os.path.exists(add + '/'+'models'):
        os.makedirs( add +'/'+'models')

def train(args, io):
    '''
    See __main__ for args parser
    '''
    writer = SummaryWriter("./log/" + args.exp_name)
    args.transformed_samples = args.transformed_samples_train 
    print(f'Using {args.dataset} dataset')
    

    samples_dist_reg = pickle.load(open( "./datasets/[spatially_partition_dataset].p", "rb" ))
    training = samples_dist_reg["Train"]    
    validation = samples_dist_reg["Valid"] 
    
    if args.dataset == 'osfa':
        in_file = 'datasets/base_dataset_name.tsv'
        sub_path = 'datasets/OSFA/train.csv'
        print(sub_path)
        df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
        label_name = 'class_label'
        output_channels = 2

    elif args.dataset == 'noZone': # this is a single model for all regions
        in_file = 'datasets/base_dataset_name.tsv'
        sub_data_reg, sub_data_norm = training[0], training[1]
        df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
        label_name = 'class_label'
        output_channels = 2

    elif args.dataset == 'place_type_1':
        in_file = 'datasets/base_dataset_name.tsv'
        sub_data_reg, sub_data_norm = training[0][args.zone], training[1][args.zone]
        # sub_path = 'datasets/place_type_3/split/train_' + str(args.train_val_index) + '.csv'
        sub_path = 'datasets/place_type_1/train.csv'
        # df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
        df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
        label_name = 'class_label'
        output_channels = 2

    elif args.dataset == 'place_type2':
        in_file = 'datasets/base_dataset_name.tsv'
        sub_data_reg, sub_data_norm = training[0][args.zone], training[1][args.zone]
        # sub_path = 'datasets/place_type_2/split/train_' + str(args.train_val_index) + '.csv'
        sub_path = 'datasets/place_type_2/train.csv'
        # df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
        df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
        
        label_name = 'class_label'
        output_channels = 2
   
    elif args.dataset == 'place_type_3':
        in_file = 'datasets/base_dataset_name.tsv'
        sub_data_reg, sub_data_norm = training[0][args.zone], training[1][args.zone]
        # sub_path = 'datasets/place_type_3/split/train_' + str(args.train_val_index) + '.csv'
        sub_path = 'datasets/place_type_3/train.csv'
        # df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
        df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
        
        label_name = 'class_label'
        output_channels = 2
    

        
    df.point_categories = df.point_categories.cat.remove_unused_categories()
    df.Sample = df.Sample.cat.remove_unused_categories()
    df.Pathology = df.Pathology.cat.remove_unused_categories()
    class_labels = list(df[label_name].cat.categories)
    num_classes = len(df['point_categories'].cat.codes.unique())
    num_samples = 0
    
    epoch = 0
    train_set = dataset.PathologyDataset(dataset=df, label_name=label_name, num_points=args.num_points, class_label = "Train", dataset_name=args.dataset,
    batch_size=args.batch_size, num_samples=num_samples, epoch = epoch, train_val_index = args.train_val_index, num_neighbors = args.num_neighbors, radius = args.radius,
    transforms=torchvision.transforms.Compose([transforms.PartitionData(n_partitions=4, n_rotations=16)]), transformed_samples=args.transformed_samples_train)
    
    sampler = dataset.get_sampler(train_set.dataset, label_name, class_labels, args.transformed_samples_train)
    
    if len(sampler)%args.batch_size>(args.batch_size/2)-1:
        drop_last = False
    else: 
        drop_last = True
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, drop_last=drop_last, sampler=sampler)
  
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args, output_channels=output_channels).to(device)
    elif args.model ==  'fcnn':
        model = FCNN(output_channels=output_channels).to(device) 
    elif args.model == 'dgcnn':
        model = DGCNN(args, output_channels=output_channels).to(device)
    elif args.model == 'sdgcnn':
        model = SpatialDGCNN(args, num_classes, output_channels).to(device)
    else:
        raise Exception("Not implemented")
    # print(str(model))
    model = nn.DataParallel(model)

    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # NOTE: GAT uses lr=0.005, weight_decay=5e-4

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss
    train_acc = 0
    best_train_acc = 0
    best_valid_acc = 0
    columns = ["train_average", "train_overall", "valid_average", "valid_overall", "train_loss", "train_times", "train_pred", "train_true"]     
    '''Output result to CSV'''
    res = pd.DataFrame(columns=columns) 


    running_loss = 0.0
    running_correct = 0.0
    counter = 0
    for epoch in range(args.epochs):
        train_average, train_losses, train_overall, train_times, valid_average, valid_overall = [], [], [], [], [], []
        start_time = time.time()
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        valid_pred = []
        valid_true = []
        batch_counter = 1
        n_total_steps = len(train_loader)
        if epoch>1:
            num_samples = len(sampler)
        for data, label in train_loader:
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                # with record_function("SAMCNet"):
            data, label = data.to(device), label.to(device).squeeze()
            batch_size = data.size()[0]
            opt.zero_grad()
            since = time.time()
            logits, _ , _ = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            now = time.time()
            # print(f'the time that takes for the batch {batch_counter} is {(now - since)/60}')
            batch_counter+=1
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            counter+=batch_size
            running_loss += loss.item()
            running_correct += (preds == label).sum().item()
            if counter%32 == 0:
                 running_loss
                 writer.add_scalar("training loss", running_loss/32, counter)
                 writer.add_scalar("training acc", running_correct/32, counter)
                 running_loss = 0.0
                 running_correct = 0    
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_f1 = metrics.f1_score(train_true, train_pred, average='weighted')
        train_precision = metrics.precision_score(train_true, train_pred, average='weighted')
        train_recall = metrics.recall_score(train_true, train_pred, average='weighted')
        
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), f'checkpoints/{args.dataset}/{args.exp_name}/models/train.t7')
            
        avg_per_class_train_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        train_times.append(time.time()-start_time)
        # print(f'the time that takes for the epoch {epoch} is {(time.time()-start_time)/60}')
        train_overall.append(train_acc)
        train_average.append(avg_per_class_train_acc)
        train_losses.append(train_loss*1.0/count)

        io.cprint(f'{datetime.now().strftime("%H:%M:%S")}: Epoch {epoch}')
        outstr = 'Train loss: %.6f, train acc: %.6f, train avg acc: %.6f, train f1 score: %.6f, train precision: %.6f, train recall: %.6f' % (
                  train_loss*1.0/count, train_acc, avg_per_class_train_acc, train_f1, train_precision, train_recall)
        io.cprint(outstr)
        torch.cuda.empty_cache()
        
        ####################
        # Validation
        ####################
        if args.dataset == 'osfa':
            in_file = 'datasets/base_dataset_name.tsv'
            sub_path = 'datasets/OSFA/val.csv'
            valid_df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
            label_name = 'class_label'
            output_channels = 2

        elif args.dataset == 'noZone':
            in_file = 'datasets/base_dataset_name.tsv'
            sub_data_reg, sub_data_norm = validation[0], validation[1]
            # sub_path = 'datasets/place_type_3/split/train_' + str(args.train_val_index) + '.csv'
            valid_df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
            # valid_df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
            label_name = 'class_label'
            output_channels = 2   

        elif args.dataset == 'place_type_1':
            in_file = 'datasets/base_dataset_name.tsv'
            sub_data_reg, sub_data_norm = validation[0][args.zone], validation[1][args.zone]
            # sub_path = 'datasets/place_type_3/split/train_' + str(args.train_val_index) + '.csv'
            sub_path = 'datasets/place_type_1/validation.csv'
            # valid_df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
            valid_df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
            # valid_df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
            label_name = 'class_label'
            output_channels = 2

        elif args.dataset == 'place_type_2':
            in_file = 'datasets/base_dataset_name.tsv'
            sub_data_reg, sub_data_norm = validation[0][args.zone], validation[1][args.zone]
            # sub_path = 'datasets/place_type_2/split/train_' + str(args.train_val_index) + '.csv'
            sub_path = 'datasets/place_type_2/validation.csv'
            # valid_df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
            # valid_df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
            valid_df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
            label_name = 'class_label'
            output_channels = 2

        elif args.dataset == 'place_type_3':
            in_file = 'datasets/base_dataset_name.tsv'
            sub_data_reg, sub_data_norm = validation[0][args.zone], validation[1][args.zone]
            # sub_path = 'datasets/place_type_3/split/train_' + str(args.train_val_index) + '.csv'
            sub_path = 'datasets/place_type_3/validation.csv'
            # valid_df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
            # valid_df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
            valid_df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
            label_name = 'class_label'
            output_channels = 2
    
        class_labels = list(valid_df[label_name].cat.categories)
        valid_df.point_categories = valid_df.point_categories.cat.remove_unused_categories()
        valid_df.Sample = valid_df.Sample.cat.remove_unused_categories()
 
        validation_set = dataset.PathologyDataset(dataset=valid_df, label_name=label_name, num_points=args.num_points, class_label = "Validation",  dataset_name=args.dataset, 
        batch_size=args.batch_size, num_samples=num_samples, epoch = epoch, train_val_index = args.train_val_index, num_neighbors = args.num_neighbors, radius = args.radius,
        transforms=torchvision.transforms.Compose([transforms.PartitionData(n_partitions=4,n_rotations=16)]), transformed_samples=args.transformed_samples_val)
        
        sampler = dataset.get_sampler(validation_set.dataset, label_name, class_labels, args.transformed_samples_val)
        if len(sampler)%args.batch_size>(args.batch_size/2)-1:
            drop_last = False
        else:
            drop_last = True
        valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, drop_last=drop_last, sampler=sampler)
 
        batch_counter = 0
        if epoch>0:
            num_samples = len(sampler)
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device).squeeze()
            logits, _ , _ = model(data)
            preds = logits.max(dim=1)[1]
            valid_true.append(label.cpu().numpy())
            valid_pred.append(preds.detach().cpu().numpy())
    
        valid_true = np.concatenate(valid_true)
        valid_pred = np.concatenate(valid_pred)
        valid_acc = metrics.accuracy_score(valid_true, valid_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(valid_true, valid_pred)
        valid_f1 = metrics.f1_score(valid_true, valid_pred, average='weighted')
        valid_precision = metrics.precision_score(valid_true, valid_pred, average='weighted')
        valid_recall = metrics.recall_score(valid_true, valid_pred, average='weighted')
        outstr = 'Validation :: valid acc: %.6f, valid avg acc: %.6f, valid f1 score: %.6f, valid precision: %.6f, valid recall: %.6f'%(valid_acc, avg_per_class_acc, valid_f1, valid_precision, valid_recall)

        if valid_acc>=best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f'checkpoints/{args.dataset}/{args.exp_name}/models/val.t7')

        valid_overall.append(valid_acc)
        valid_average.append(avg_per_class_acc)
        csv = {
        'train_overall':  train_overall,
        'train_f1': train_f1,
        'train_precision': train_precision,
        'train_recall': train_recall, 
        'valid_overall': valid_overall,
        'valid_f1': valid_f1,
        'valid_precision': valid_precision,
        'valid_recall': valid_recall,
        'train_loss':  train_losses,
        'train_times': train_times,
        }

        res = res.append(csv, ignore_index=True)
        # saving the dataframe 
        res.to_csv(args.save_train_results +  str(args.exp_name) + "/results.csv")

        io.cprint(outstr)
        torch.cuda.empty_cache()

def test(args, io):
    samples_dist_reg = pickle.load(open( "./datasets/[spatially_partition_dataset].p", "rb" ))
    testing = samples_dist_reg["Test"]  

    if args.dataset == 'osfa':
        in_file = 'datasets/base_dataset_name.tsv'
        sub_path = 'datasets/OSFA/test.csv'
        # sub_path = 'datasets/OSFA/test.csv'
        print(sub_path)
        df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
        print(dict(enumerate(df['point_categories'].cat.categories)))
        label_name = 'class_label'
        output_channels = 2

    elif args.dataset == 'noZone': # this is a single model for all regions
        in_file = 'datasets/base_dataset_name.tsv'
        sub_data_reg, sub_data_norm = testing[0], testing[1]
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
        label_name = 'class_label'
        output_channels = 2

    elif args.dataset == 'place_type_1':
        in_file = 'datasets/base_dataset_name.tsv'
        sub_data_reg, sub_data_norm = testing[0][args.zone], testing[1][args.zone]
        # sub_path = 'datasets/place_type_3/split/train_' + str(args.train_val_index) + '.csv'
        sub_path = 'datasets/place_type_1/test.csv'
        # df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
        df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
        label_name = 'class_label'
        output_channels = 2

    elif args.dataset == 'place_type_2':
        in_file = 'datasets/base_dataset_name.tsv'
        sub_path = 'datasets/place_type_2/test.csv'
        # sub_data_reg, sub_data_norm = testing[0][args.zone], testing[1][args.zone]
        df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
        print(dict(enumerate(df['point_categories'].cat.categories)))
        label_name = 'class_label'
        output_channels = 2


    elif args.dataset == 'place_type_3':
        in_file = 'datasets/base_dataset_name.tsv'
        sub_path = 'datasets/place_type_3/test.csv'
        # sub_data_reg, sub_data_norm = testing[0][args.zone], testing[1][args.zone]
        df = dataset.read_dataset(in_file, sub_path = sub_path, zonal=True)
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_reg)
        # df = dataset.read_dataset(in_file, sub_path = None, sub_data=sub_data_norm)
        print(dict(enumerate(df['point_categories'].cat.categories)))
        label_name = 'class_label'
        output_channels = 2

    
    df.point_categories = df.point_categories.cat.remove_unused_categories()
    df.Sample = df.Sample.cat.remove_unused_categories()
    num_classes = len(df.point_categories.cat.codes.unique())    
    test_set = dataset.PathologyDataset(dataset=df, label_name=label_name, num_points=args.num_points, class_label = "Test", dataset_name=args.dataset,
    batch_size=args.test_batch_size, num_samples=0, epoch = 0, train_val_index = args.train_val_index,  num_neighbors = args.num_neighbors, 
    radius = args.radius)
    
    if len(test_set)%args.test_batch_size>args.test_batch_size/2 :
        drop_last = False
    else:
        drop_last = False
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, drop_last=drop_last)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args, output_channels=output_channels).to(device)
    elif args.model ==  'fcnn':
        model = FCNN(output_channels=output_channels).to(device) 
    elif args.model ==  'dgcnn':
        model = DGCNN(args, output_channels=output_channels).to(device) 
    else:
        model = SpatialDGCNN(args, num_classes, output_channels).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    test_true = []
    test_pred = []
    prioritization_4 =  {}
    point_pairs_add = {}
    count = 0
    batch_counter = 1
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        logits, prior_4, pointpair = model(data)
        '''
        Extracting features for other classifiers
        '''
        prioritization_4[count] =  prior_4.cpu()
        point_pairs_add[count] =  pointpair.cpu()
        batch_counter+=1
        count+=1
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='weighted')
    test_precision = metrics.precision_score(test_true, test_pred, average='weighted')
    test_recall = metrics.recall_score(test_true, test_pred, average='weighted')
        

    test_true = []
    test_pred = []
    outstr = 'test :: test acc: %.6f, test avg acc: %.6f, test f1 score: %.6f, test precision: %.6f, test recall: %.6f'%(test_acc, avg_per_class_acc, test_f1, test_precision, test_recall)
    io.cprint(outstr)
    pickle.dump(prioritization_4, open(args.save_features + "prioritization_4.p", "wb" ))
    pickle.dump(point_pairs_add, open(args.save_features + "point_pairs.p", "wb" ))
        
if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Spatial DGCNN')
    
    parser.add_argument('--transformed_samples_train', type=int, default=3, metavar='N',
                        help='Num of data transformation')
    parser.add_argument('--transformed_samples_val', type=int, default=3, metavar='N',
                        help='Num of data transformation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--zone', type=str, default='place_type_2', metavar='N',
                        choices=['all', 'place_type_1', 'place_type_2', 'place_type_3'],
                        help='Zone to use, [all, place_type_1, place_type_2, place_type_3]')
    parser.add_argument('--dataset', type=str, default='place_type_2', metavar='N',
                        choices=['osfa', 'place_type_1', 'place_type_2', 'place_type_3', 'noZone'],
                        help='Dataset to use, [osfa, place_type_1, place_type_2, place_type_3]')
    parser.add_argument('--model', type=str, default='sdgcnn', metavar='N',
                        choices=['pointnet', 'fcnn',  'dgcnn', 'sdgcnn'],
                        help='Model to use, [pointnet, dgcnn, sdgcnn]') 
    parser.add_argument('--use_pe', type=bool,  default=False,
                        help='use positional encoding')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--radius', type=int, default=100, metavar='N',
                        help='neighborhood distance')
    parser.add_argument('--num_neighbors', type=int, default=5, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=True,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='num of attn heads to use. Set to 0 for no attn')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=5, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='checkpoints/[model_path]', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--PE_dim', type=float, default=4, metavar='N',
                        help='If use_PE True, output dimmension of positional encoding (if use_pe fasle) this should be 4')
    parser.add_argument('--save_features', type=str, default='./checkpoints/[save_features_path]', metavar='N',
                        help='Save extracted features path')
    parser.add_argument('--save_train_results', type=str, default='./checkpoints/[save_train&val_results]/', metavar='N',
                        help='save training results (e.g., loss, acc, ... )')
    parser.add_argument('--train_val_index', type=int, default=1) 
    args = parser.parse_args()
 
    _init_()

    io = IOStream('checkpoints/' + args.dataset + "/" + args.exp_name + "/" + args.exp_name + '.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io) # Not implemented (yet)
