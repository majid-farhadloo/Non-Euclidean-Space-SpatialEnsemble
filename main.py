# {0: 'B Cell', 1: 'Cytotoxic T Cell', 2: 'Helper T Cell', 3: 'Macrophage', 4: 'Monocyte', 5: 'Neutrophil', 6: 'Plasma Cell', 7: 'Regulatory T Cell', 8: 'Tumor Cell', 9: 'Vasculature'}
from __future__ import print_function
from audioop import avg
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import PointNet, DGCNN, FCNN, Point_Transformer
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

def _init_(exp_name):
    add =  './checkpoints/' + args.place_type + "/" + exp_name
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(add):
        os.makedirs(add)
        print(add) 
    if not os.path.exists(add + '/'+'models'):
        os.makedirs( add +'/'+'models')
        print(add +'/'+'models') 

def load_model(args, output_channels, num_classes):
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args, output_channels=output_channels).to(device)
    elif args.model ==  'fcnn':
        model = FCNN(output_channels=output_channels).to(device) 
    elif args.model == 'dgcnn':
        model = DGCNN(args, output_channels=output_channels).to(device)
    elif args.model == 'samcnet':
        model = SpatialDGCNN(args, num_classes, output_channels).to(device)
    elif args.model == 'pointTransformers':
        ## Hyperparameters
        config = {'num_points' : 1024,
            'batch_size': 16,
            'use_normals': False,
            'optimizer': 'adam',
            'lr': 0.001,
            'decay_rate': 1e-06,
            'epochs': 200,
            'num_classes': 2,
            'dropout': 0.4,
            'M': 4,
            'K': 15,
            'd_m': 512,
            }
        model = Point_Transformer(config).to(device)
    else:
        raise Exception("Not implemented")
    # print(str(model))
    return model

def training_iter(args, data_loaders_train, data_loaders_valid, io, epochs, num_classes = None, output_channels = 2, save_train_results = None, exp_name = "global", pretrain_path = None, pretrain_status = False):
    
    add =  f'checkpoints/{args.place_type}/{exp_name}/models/'
    if not os.path.exists(add):
        os.makedirs(add)
        
    writer = SummaryWriter("./log/" + args.exp_name_global)
    model = load_model(args, output_channels, num_classes)
    model = nn.DataParallel(model)
        
    "Freezing model parameters"
    if pretrain_status:
            model.load_state_dict(torch.load(pretrain_path))
            # model_modules = list(model.module.children())[:4]
            model_modules = list(model.module.children())[:args.num_frozen_layers]
            for module in model_modules:
                for parameters in module.parameters():
                    parameters.requires_grad = False
    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # NOTE: GAT uses lr=0.005, weight_decay=5e-4

    scheduler = CosineAnnealingLR(opt, epochs, eta_min=args.lr)
    criterion = cal_loss
    train_acc, best_train_acc, best_valid_acc = 0, 0, 0
    columns = ["train_average", "train_overall", "valid_average", "valid_overall", "train_loss", "train_times", "train_pred", "train_true"]     
    '''Output result to CSV'''
    res = pd.DataFrame(columns=columns) 
    running_loss, running_correct, counter = 0.0, 0.0, 0

    for epoch in range(epochs):
        train_average, train_losses, train_overall, train_times, valid_average, valid_overall = [], [], [], [], [], []
        start_time = time.time()
        scheduler.step()
        ####################
        # Train model
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred, train_true, valid_pred, valid_true = [], [], [], []
        for data_batch in data_loaders_train:
            if args.weighted_distance:
                distance_id = data_batch.dataset.dataset.lr_id.unique().item()
                opt.param_groups[0]["lr"] = args.lr/distance_id
            for data, label in data_batch:
                data, label = data.to(device), label.to(device).squeeze()
                batch_size = data.size()[0]
                opt.zero_grad()
                data = data[:,:,:3]
                if args.model == "pointTransformers":
                    data = data.permute(0,2,1)
                # print(data, label)
                logits, _, _ = model(data)
                # print(logits)
                # sys.exit(-1)
                loss = criterion(logits, label)
                loss.backward()
                opt.step()
                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                # print("label: ", label.cpu().numpy())
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
                counter+=batch_size
                running_loss += loss.item()
                running_correct += (preds == label).sum().item()
                if counter%64 == 0:
                    running_loss
                    writer.add_scalar("training loss", running_loss/32, counter)
                    writer.add_scalar("training acc", running_correct/32, counter)
                    running_loss, running_correct = 0.0, 0.0
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        train_f1 = metrics.f1_score(train_true, train_pred, average='weighted')
        train_precision = metrics.precision_score(train_true, train_pred, average='weighted')
        train_recall = metrics.recall_score(train_true, train_pred, average='weighted')
        
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), f'checkpoints/{args.place_type}/{exp_name}/models/train.t7')

        avg_per_class_train_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        train_times.append(time.time()-start_time)
        train_overall.append(train_acc)
        train_average.append(avg_per_class_train_acc)
        train_losses.append(train_loss*1.0/count)

        io.cprint(f'{datetime.now().strftime("%H:%M:%S")}: Epoch {epoch}')
        outstr = 'Train loss: %.6f, train acc: %.6f, train avg acc: %.6f, train f1 score: %.6f, train precision: %.6f, train recall: %.6f' % (
                  train_loss*1.0/count, train_acc, avg_per_class_train_acc, train_f1, train_precision, train_recall)
        io.cprint(outstr)
        torch.cuda.empty_cache()  

        for data_batch in data_loaders_valid:
            if args.weighted_distance:
                distance_id = data_batch.dataset.dataset.lr_id.unique().item()
                opt.param_groups[0]["lr"] = args.lr/distance_id
            for data, label in data_batch:
                data, label = data.to(device), label.to(device).squeeze()
                data = data[:,:,:3]
                if args.model == "pointTransformers":
                    data = data.permute(0,2,1)
                logits, _, _ = model(data)
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
            torch.save(model.state_dict(), f'checkpoints/{args.place_type}/{exp_name}/models/val.t7')

        valid_overall.append(valid_acc)
        valid_average.append(avg_per_class_acc)
        csv = {
        'train_overall':  train_overall, 'train_f1': train_f1, 'train_precision': train_precision,'train_recall': train_recall, 
        'valid_overall': valid_overall, 'valid_f1': valid_f1, 'valid_precision': valid_precision, 'valid_recall': valid_recall,
        'train_loss':  train_losses,'train_times': train_times,}
        res = res.append(csv, ignore_index=True)
        res.to_csv(save_train_results + "/results.csv")
        io.cprint(outstr)
        torch.cuda.empty_cache()   
    print("Training is finished!")
    model_path = f'checkpoints/{args.place_type}/{exp_name}/models/train.t7'
    return model, model_path

def test_iter(args, data_loaders_test, io, num_classes = None, output_channels = 2, test_res_local = None, model_path = None, save_feats_path = None, feats_extract = False):
    # columns = ["test_prob", "test_pred", "test_true"]     
    # '''Output result to CSV'''
    # res = pd.DataFrame(columns=columns)
    model = load_model(args, output_channels, num_classes)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    test_acc = 0.0
    # test_true , test_pred = [], []
    fe_256, fe_512 = {}, {}
    for epoch in range(1):
        test_prob, test_pred, test_true = [], [], []
        count = 0
        for data_batch in data_loaders_test:
            for data, label in data_batch:
                data, label = data.to(device), label.to(device).squeeze()
                if args.model == "pointTransformers":
                    data = data.permute(0,2,1)
                '''
                Extracting features
                '''
                if feats_extract:
                    logits, fc_512, fc_256 = model(data)
                    fe_512[count]= fc_512.detach().cpu()
                    fe_256[count]= fc_256.detach().cpu()
                    count+=1
                else:
                    logits, _, _ = model(data)
                    probabilities = F.softmax(logits, dim=1)
                preds = logits.max(dim=1)[1]
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
                test_prob.append(probabilities.max(dim=1)[0].detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_prob = np.concatenate(test_prob)

        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_f1 = metrics.f1_score(test_true, test_pred, average='weighted')
        test_precision = metrics.precision_score(test_true, test_pred, average='weighted')
        test_recall = metrics.recall_score(test_true, test_pred, average='weighted')
        

        print("test_pred: -->", len(test_pred))

        csv = {'test_prob':  test_prob, 'test_pred': test_pred, 'test_true': test_true}
        pickle.dump(csv, open(test_res_local + 'result.p', "wb" ))
        # res = res.append(csv, ignore_index=True)
        # res.to_csv(test_res_local + "/results.csv")

        outstr = 'test :: test acc: %.6f, test avg acc: %.6f, test f1 score: %.6f, test precision: %.6f, test recall: %.6f'%(test_acc, avg_per_class_acc, test_f1, test_precision, test_recall)
        io.cprint(outstr)
        if feats_extract:
            pickle.dump(fe_512, open(save_feats_path + "_fc_512.p", "wb" ))
            pickle.dump(fe_256, open(save_feats_path + "_fc_256.p", "wb" ))
        
    
    

def prep_data(args, dataset_path = 'datasets/BestClassification_July2021_14Samples.tsv', place_type = "Interface", dataset_type = "train", num_transformed_sample = 1):
    
    if args.weighted_distance:
        sub_path = 'datasets/' + place_type + '/' + dataset_type + '_target.csv'
        print(sub_path)
        df, data_samples = dataset.read_dataset(dataset_path, sub_path = sub_path, dataset_type = "weighted_distance")
    else:
        sub_path = 'datasets/' + place_type + '/' + dataset_type + '.csv'
        print(sub_path)
        df, data_samples = dataset.read_dataset(dataset_path, sub_path = sub_path, dataset_type = dataset_type)
        
    label_name = 'Status'
    output_channels = 2

    df.Phenotype = df.Phenotype.cat.remove_unused_categories()
    df.Sample = df.Sample.cat.remove_unused_categories()
    df.Pathology = df.Pathology.cat.remove_unused_categories()
    class_labels = list(df[label_name].cat.categories)

    num_classes = len(df['Phenotype'].cat.codes.unique())
    if dataset_type == "train" or dataset_type == "val":
        if args.weighted_distance:
            lr_ids = df.lr_id.unique()
            data_loader_set = []  
            for lr_id in lr_ids:
                sample_data = df[df.lr_id == lr_id]
                data_points = data_samples[data_samples.lr_id == lr_id]
                data_set = dataset.PathologyDataset(dataset=sample_data, label_name=label_name, num_points=args.num_points,
                batch_size=args.batch_size, train_val_index = args.train_val_index, 
                transformed_samples= num_transformed_sample, dataset_type="weighted_distance",
                transforms=torchvision.transforms.Compose([transforms.PartitionData(n_partitions=4, n_rotations=16)]))
                
                sampler = dataset.get_sampler(data_set.dataset, label_name, class_labels, data_points, args.transformed_samples_train)
                
                if len(sampler)%args.batch_size>(args.batch_size/2)-1:
                    drop_last = False
                else: 
                    drop_last = True
                dl = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, drop_last=drop_last, sampler=sampler)
                data_loader_set.append(dl)
            
        else:
                data_set = dataset.PathologyDataset(dataset=df, label_name=label_name, num_points=args.num_points, 
                batch_size=args.batch_size, train_val_index = args.train_val_index, 
                transformed_samples= num_transformed_sample, dataset_type=place_type,
                transforms=torchvision.transforms.Compose([transforms.PartitionData(n_partitions=4, n_rotations=16)]))
            
                sampler = dataset.get_sampler(data_set.dataset, label_name, class_labels, data_samples, args.transformed_samples_train)
            
                
                if len(sampler)%args.batch_size>(args.batch_size/2)-1:
                    drop_last = False
                else: 
                    drop_last = True

                dl = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, drop_last=drop_last, sampler=sampler)
                data_loader_set = [dl]
    else:
        test_set = dataset.PathologyDataset(dataset=df, label_name=label_name, num_points=args.num_points, batch_size=args.test_batch_size, 
                    train_val_index = 1, dataset_type=place_type) 
        print("test_set: -->", len(test_set))       
        dl = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, drop_last=False)
        data_loader_set = [dl]          

    return data_loader_set, num_classes, output_channels

def pretrain_learner(args, io):
    
    data_loaders_train, num_classes, output_channels = prep_data(args, place_type="OSFA", dataset_type="train", num_transformed_sample= args.transformed_samples_train)
    data_loaders_valid, _, _ = prep_data(args, place_type="OSFA", dataset_type="val", num_transformed_sample = args.transformed_samples_val)
    
    pretrained_model, model_path = training_iter(args = args, data_loaders_train = data_loaders_train, data_loaders_valid = data_loaders_valid, io = io, epochs = args.epochs_global, num_classes = num_classes, output_channels = output_channels, save_train_results = args.train_res_global, exp_name = args.exp_name_global)
    
    return pretrained_model, model_path


def local_learner(args, pretrained_model_path, io):
    # re-train the network for deep layer based on the place type (e.g., 1, 2, 3) #
    
    data_loaders_train, _, output_channels = prep_data(args, place_type = "Interface", dataset_type="train", num_transformed_sample= args.transformed_samples_train)
    data_loaders_valid, _, _ = prep_data(args, place_type = "Interface", dataset_type="val", num_transformed_sample = args.transformed_samples_val)
    num_classes = 8

    
    place_specifc_model, place_specifc_model_path = training_iter(args = args, data_loaders_train = data_loaders_train, data_loaders_valid = data_loaders_valid, io = io, 
                                                                  num_classes = num_classes, output_channels = output_channels,
                                                                save_train_results=args.train_res_local, exp_name = args.exp_name_local, 
                                                                pretrain_path = global_model_path, pretrain_status = False, epochs=args.epochs_local)
    return place_specifc_model, place_specifc_model_path

def test(args, io):
    data_loaders_test, num_classes, output_channels = prep_data(args, place_type=args.place_type, dataset_type="test")

    model_path = args.local_learner_path

    test_iter(args = args, data_loaders_test = data_loaders_test, io = io, 
              num_classes = 8, output_channels = 2, test_res_local = args.test_res_local, model_path = model_path, save_feats_path = None, feats_extract = False)
    

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Spatial DGCNN')
    
    parser.add_argument('--transformed_samples_train', type=int, default=3, metavar='N',
                        help='Num of data transformation')
    parser.add_argument('--transformed_samples_val', type=int, default=3, metavar='N',
                        help='Num of data transformation')
    parser.add_argument('--place_type', type=str, default='Normal', metavar='N',
                        choices=['Normal', 'Interface', 'Tumor', 'OSFA'], #place_type_1, place_type_2, place_type_3
                        help='to use data generation by clusters or regular data')
 
    parser.add_argument('--exp_name_global', type=str, default='global_wdlr', metavar='N',
                        help='Name of the experiment')
    
    parser.add_argument('--exp_name_local', type=str, default='local_wdlr', metavar='N',
                        help='Name of the experiment') 

    parser.add_argument('--pretrain_status', type=bool, default=False, metavar='N',
                        help='if a pretrained model is already exist')

    parser.add_argument('--model', type=str, default='samcnet', metavar='N',
                        choices=['pointnet', 'fcnn',  'dgcnn', 'pointTransformers',  'samcnet'],
                        help='Model to use, [pointnet, dgcnn, pointTransformers, samcnet]') 
    
    parser.add_argument('--weighted_distance', type=bool,  default=False,
                        help='use weighted-distance learning rate')
    
    parser.add_argument('--use_pe', type=bool,  default=True,
                        help='use positional encoding')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs_global', type=int, default=200, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--epochs_local', type=int, default=150, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--radius', type=int, default=100, metavar='N',
                        help='neighborhood distance')
    parser.add_argument('--num_neighbors', type=int, default=5, metavar='N',
                        help='number of neighbors to select')
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
    parser.add_argument('--eval', type=bool,  default=False,
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
    parser.add_argument('--PE_dim', type=float, default=32, metavar='N',
                        help='If use_PE True, output dimmension of positional encoding (if use_pe fasle) this should be 4')
    
    parser.add_argument('--num_frozen_layers', type=int, default=4, metavar='N',
                        help='If exists a pre-trained model, how many layers should be frozen for fine-tuning')
    
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    
    parser.add_argument('--train_res_global', type=str, default='', metavar='N',
                        help='save training results for global model (e.g., loss, acc, ... )')
    parser.add_argument('--train_res_local', type=str, default='', metavar='N',
                        help='save training results for local model (e.g., loss, acc, ... )')
    
    parser.add_argument('--test_res_local', type=str, default='', metavar='N',
                        help='save training results for local model (e.g., loss, acc, ... )')
    
    parser.add_argument('--local_learner_path', type=str, default='', metavar='N',
                        help='fine-tuned model path of local model')
    
    parser.add_argument('--train_val_index', type=int, default=1) 
    args = parser.parse_args()
 
    _init_(args.exp_name_global)
    _init_(args.exp_name_local)
    io = IOStream('checkpoints/' + args.place_type + "/" + args.exp_name_global + "/" + args.exp_name_global + '.log')
    # io.cprint(str(args))
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        if args.pretrain_status:
            pretrained_model, global_model_path = pretrain_learner(args=args, io=io)
            io = IOStream(f'checkpoints/' + args.place_type + "/" + args.exp_name_local + "/" + args. exp_name_local + '.log')
            io.cprint(str(args))
            place_specifc_model, local_model_path = local_learner(args, global_model_path, io=io)
        else:
            io = IOStream(f'checkpoints/' + args.place_type + "/" + args.exp_name_local + "/" + args. exp_name_local + '.log')
            io.cprint(str(args))
            global_model_path = args.local_learner_path
            place_specifc_model, local_model_path = local_learner(args, global_model_path, io=io)
             
    else:
        io = IOStream(f'checkpoints/' + args.place_type + "/" + args.exp_name_local + "/" + args.exp_name_local + '.log')
        io.cprint(str(args))
        test(args, io) # Not implemented (yet)

        # participation_ratio_files = pickle.load( open( prs + "BMS_AIM1_PRs.p", "rb"))
    