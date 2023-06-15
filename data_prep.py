import os
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.spatial import KDTree
from sklearn.preprocessing import Normalizer
import random
import pickle
import warnings
import scipy.stats as stats
warnings.filterwarnings("ignore")

def clean(df):
    df = df.reset_index()
    df = df.drop(columns=["index"])
    return df

def update_dic(dic,key,new_curr):
    curr = dic[key]
    curr = curr.append(new_curr)
    dic[key] = curr
    return dic

def read_tsv(in_file, set_categories=True, sep=','): # read dataset and set necessary column as categorical values 
    df = pd.read_csv(in_file, sep=sep, header=0, low_memory=False)
    if set_categories:
        # Set necessary columns as categorical to later retrieve distinct category codes
        df.Sample = pd.Categorical(df.Sample)
    return df

class SplitData():
    def __init__(self, place_types_coordinates_table, train_ratio):
        self.place_types_coordinates_table = place_types_coordinates_table
        self.train_ratio = train_ratio
    def create_train_val_test(self, df):
        
        training = {"place_type_1": pd.DataFrame(), "place_type_2": pd.DataFrame(), "place_type_3": pd.DataFrame()}
        validation = {"place_type_1": pd.DataFrame(), "place_type_2": pd.DataFrame(), "place_type_3": pd.DataFrame()}
        testing = {"place_type_1": pd.DataFrame(), "place_type_2": pd.DataFrame(), "place_type_3": pd.DataFrame()}

        train_all = pd.DataFrame()
        val_all = pd.DataFrame()
        test_all = pd.DataFrame()
        regs = df.region.unique()
        samples = df.Sample.unique()
        for s in samples:
            ts = df[df.Sample == s] # a selected ts (tissue sample)
            for i in range(len(regs)):
                region  = ts[ts.region == regs[i]] # all samples for a place_types (e.g., interface) in a tissue slide
                region = region.reset_index()
                region = region.drop(columns=["index"])
                train = set(random.sample(range(len(region)),int(self.train_ratio*len(region))))
                test = set(range(len(region))) - train
                val = set(random.sample(train,int(0.25*len(train))))
                train = train - val
                training = update_dic(training, regs[i], region.iloc[list(train)])
                validation = update_dic(validation, regs[i], region.iloc[list(val)])
                testing = update_dic(testing, regs[i], region.iloc[list(test)])

        for i in range(len(regs)):
            train_data, val_data, test_data = training[regs[i]], validation[regs[i]], testing[regs[i]]
            train_data, val_data, test_data = clean(train_data), clean(val_data), clean(test_data)

            train_all, val_all, test_all= train_all.append(train_data),  val_all.append(val_data), test_all.append(test_data)

        train_all, val_all, test_all = clean(train_all), clean(val_all), clean(test_all)
        return train_all, val_all, test_all
if __name__ == "__main__":
    in_file_place_types_locs = './datasets/[place_types_coordinates_table].csv'
    df = read_tsv(in_file_place_types_locs) # read dataset
    split_data = SplitData(in_file_place_types_locs, 0.8)
    train, val, test = split_data.create_train_val_test(df)
    train.to_csv("./datasets/place_typesLocationsTrain.csv")
    val.to_csv("./datasets/place_typesLocationsVal.csv")
    test.to_csv("./datasets/place_typesLocationsTest.csv")
