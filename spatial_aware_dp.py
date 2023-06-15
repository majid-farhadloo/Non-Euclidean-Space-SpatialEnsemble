import pandas as pd
import numpy as np
import torch
from scipy.spatial import distance
from scipy.spatial import KDTree
from sklearn.preprocessing import Normalizer
import pickle
import warnings
import scipy.stats as stats

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    
def normalize(df, option=2): # normalize X and Y coordinates if self.normalized = True
    if option == 1:
        # option 1 z-score normalization (mean =0, std = 1)
        df["norm_x"]=(df.X-df.X.mean())/df.X.std()
        df["norm_y"]=(df.Y-df.Y.mean())/df.Y.std()

    elif option == 2:
        # option 2 min max normalization (min = 0, max = 1)
        df["norm_x"]=(df.X-df.X.min())/(df.X.max()-df.X.min())
        df["norm_y"]=(df.Y-df.Y.min())/(df.Y.max()-df.Y.min())
    else: 
        ########## option 3 needs to be fixed ###################
        # option 3 normalizer function rescales individual values until its l2 or l1 norm is equal to one.
        x_transformer = Normalizer().fit(df.X)
        y_transformer = Normalizer().fit(df.Y)
        df["norm_x"] = x_transformer.transform(df.X)
        df["norm_y"] = y_transformer.transform(df.Y)
    return df

def flattern_list(l):
    flat_list = [item for sublist in l for item in sublist]
    return list(set(flat_list))

class DrivedModelLoc():
    def __init__(self, place_types_coordinates_table, based_on_reg = True, norm = True):
    # def __init__(self, dataset, num_points=None, transforms=None, transformed_samples=1, label_name='Margin', plot_data=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.place_types_coordinates_table = place_types_coordinates_table
        self.norm = norm # this is for normalized relative distance based on FOV locations.
        self.based_on_reg = based_on_reg
    
    def const_model_cal(self, df):
        min_x,max_x,min_y,max_y = df.X.min(), df.X.max(), df.Y.min(), df.Y.max()
        n_min_x, n_max_x, n_min_y, n_max_y = df.norm_x.min(), df.norm_x.max(), df.norm_y.min(), df.norm_y.max()
        reg_model_loc = [(min_x+max_x)/2, (min_y+max_y)/2]
        norm_model_loc = [(n_min_x+n_max_x)/2, (n_min_y+n_max_y)/2]
        
        return [reg_model_loc[0],reg_model_loc[1], norm_model_loc[0],norm_model_loc[1]]
        
    
    def const_model_loc(self):
        regional_model_locs = {"place_type_1": [], "place_type_2": [], "place_type_3": []}
                
        
        df = read_tsv(self.place_types_coordinates_table) # read dataset
        
        if self.norm: 
            df = normalize(df) # normalize x,y coords if needed! 
            
        based_on_reg = self.based_on_reg
        regions = list(df.region.unique())
        if based_on_reg:
            for reg in regions:
                place_categpry  = df[df.region == reg]
                regional_model_locs[reg] = self.const_model_cal(place_categpry)
            return regional_model_locs
        else:    
            return self.const_model_cal(df)
        
    def derived_model_loc(self, based_on_reg = True):
        based_on_reg = self.based_on_reg
        
        global_model_loc = []
        global_derived_dist = []
        regional_model_locs = {"place_type_1": [], "place_type_2": [], "place_type_3": []}
        reg_derived_dist = {"place_type_1": [], "place_type_2": [], "place_type_3": []}


        df = read_tsv(self.place_types_coordinates_table) # read dataset
        
        if self.norm: 
            df = normalize(df) # normalize x,y coords if needed! 
        
        samples = df.Sample.unique() # this is tissue sample from different ROI has been selected from
        for s in samples:
            ts = df[df.Sample == s] # a selected ts (tissue sample)
            if based_on_reg:
                regs = sorted(ts.region.unique())
                stat = ts.groupby("region").mean().reset_index()
                for i in range(len(regs)):
                    temp = stat[stat.region == regs[i]].values[0,1:] # X,Y, x_norm, y_norm
                    curr = regional_model_locs[regs[i]]
                    curr.append(temp)
                    regional_model_locs[regs[i]] = curr
                    centroid_reg, centroid_norm = [(temp[0],temp[1])], [(temp[2],temp[3])]
            else:
                means = ts.groupby("region").mean().mean().to_numpy()
                global_model_loc.append(means)
                centroid_reg, centroid_norm = [(means[0],means[1])], [(means[2],means[3])]

        if based_on_reg:
            regs = list(regional_model_locs.keys())
            for i in range(len(regs)):
                curr = regional_model_locs[regs[i]]
                curr = np.stack(curr, axis = 0)
                curr = np.mean(curr, axis = 0)
                regional_model_locs[regs[i]] = curr # X,Y, x_norm, y_norm
                res = regional_model_locs
                
                temp = df[df.region == regs[i]]
                coords_reg = temp[["X","Y"]].to_numpy()
                coords_norm = temp[["norm_x","norm_x"]].to_numpy()
                centroid_reg, centroid_norm = [(curr[0],curr[1])], [(curr[2],curr[3])]
                
                d_reg = distance.cdist(centroid_reg, coords_reg, 'euclidean')
                d_norm = distance.cdist(centroid_norm, coords_norm, 'euclidean')
                # Performing normality test
                _, p_value_reg = stats.normaltest(d_reg[0])
                _, p_value_norm = stats.normaltest(d_norm[0])
                if p_value_reg < 0.05:
                    d_reg = np.mean(d_reg)
                else:
                    d_reg = np.quantile(d_reg, 0.75)
                if p_value_norm < 0.05:
                    d_norm = np.mean(d_norm)
                else:
                    d_norm = np.quantile(d_norm, 0.75)
                
                curr_d = [d_reg, d_norm]
                reg_derived_dist[regs[i]] = curr_d
            res_d = reg_derived_dist     
        else:
            curr = np.stack(global_model_loc, axis = 0)
            curr = np.mean(curr, axis = 0)
            res = curr

            coords_reg = df[["X","Y"]].to_numpy()
            coords_norm = df[["norm_x","norm_x"]].to_numpy()
            centroid_reg, centroid_norm = [(curr[0],curr[1])], [(curr[2],curr[3])]
           
            d_reg = distance.cdist(centroid_reg, coords_reg, 'euclidean')
            d_norm = distance.cdist(centroid_norm, coords_norm, 'euclidean')
            # Performing normality test
            _, p_value_reg = stats.normaltest(d_reg[0])
            _, p_value_norm = stats.normaltest(d_norm[0])
            if p_value_reg < 0.05:
                d_reg = np.mean(d_reg)
            else:
                d_reg = np.quantile(d_reg, 0.5)
            if p_value_norm < 0.05:
                d_norm = np.mean(d_norm)
            else:
                d_norm = np.quantile(d_norm, 0.5)
            

            curr_d = [d_reg, d_norm]
            res_d = curr_d
        return res, res_d
    
class SampleSelection():
    def __init__(self, place_types_coordinates_table, k, const_model_loc = False, based_on_distance = True, based_on_reg = False, norm = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.place_types_coordinates_table = place_types_coordinates_table 
        self.k = k  # his is for K-nn nearest algorithm to select samples. 
        self.norm = norm # this is for normalized relative distance based on FOV locations.  
        self.based_on_region = based_on_reg # to select one model or multiple based on number of classes! 
        self.based_on_distance = based_on_distance # use k-nearest algorithm or select samples based on distance
        self.const_model_loc = const_model_loc
        self.derivedModelLoc = DrivedModelLoc(self.place_types_coordinates_table, self.based_on_region, self.norm)

    def derived_dist_select_samples(self,fov,model_loc_reg,model_loc_norm,dist_reg,dist_norm): 
        # get corresponding samples for each model location given 
        coords_reg = fov[["X","Y"]].to_numpy()
        coords_norm = fov[["norm_x","norm_x"]].to_numpy()
        d1 = distance.cdist(model_loc_reg, coords_reg, 'euclidean')
        d2 = distance.cdist(model_loc_norm, coords_norm, 'euclidean')
        selected1 = fov.iloc[np.where(np.any(d1<dist_reg, axis=0))]
        selected2 = fov.iloc[np.where(np.any(d2<dist_norm, axis=0))]
        
        fovs1 = list(selected1["s_fov"])
        fovs2 = list(selected2["s_fov"])
        return fovs1, fovs2

    def derived_K_select_samples(self,fov,model_loc_reg,model_loc_norm,k): 
        # get corresponding samples for each model location given
        fov = fov.reset_index() 
        coords_reg = fov[["X","Y"]].to_numpy()
        coords_norm = fov[["norm_x","norm_x"]].to_numpy()
        coords_reg_t = KDTree(coords_reg)
        coords_norm_t = KDTree(coords_norm)

        _, reg_samples = coords_reg_t.query(model_loc_reg, k=k)
        reg_samples = np.where(reg_samples[0]<len(fov), reg_samples[0], reg_samples[0][0])
        _, norm_samples = coords_norm_t.query(model_loc_norm, k=k)
        norm_samples = np.where(norm_samples[0]<len(fov), norm_samples[0], norm_samples[0][0])

        if len(fov)>0:
            selected1 = fov.iloc[reg_samples]
            selected2 = fov.iloc[norm_samples]
            
            fovs1 = list(selected1["s_fov"])
            fovs2 = list(selected2["s_fov"])
            return fovs1, fovs2
        return [], []
    
    def update_list_dic(self, reg_dist_selected_samples, norm_dist_selected_samples, fovs1, fovs2, reg):
        curr = reg_dist_selected_samples[reg]
        curr.append(fovs1)
        reg_dist_selected_samples[reg] = curr

        curr = norm_dist_selected_samples[reg]
        curr.append(fovs2)
        norm_dist_selected_samples[reg] = curr
    
        return reg_dist_selected_samples, norm_dist_selected_samples
    
    def select_samples(self, k, based_on_region = False, based_on_distance = True):
        k = self.k # k nearest samples that needs to be selected
        based_on_distance = self.based_on_distance # if we choose fixed-neighborhood distance or not! 
        fov_coordinates = read_tsv(self.place_types_coordinates_table) # read fov coordinates table
        if self.norm:
            fov_coordinates = normalize(fov_coordinates)

        based_on_region = self.based_on_region # if we have a single model or multiple model (one model per region)
        if not self.const_model_loc:
            model_loc, model_dist = self.derivedModelLoc.derived_model_loc(self.based_on_region) # get model location and derived distances (both regular and normalized (rescaled))
        else:
            model_loc = self.derivedModelLoc.const_model_loc()
        
        reg_dist_ss_regional = {"place_type_1": [], "place_type_2": [], "place_type_3": []} # dictionary for fixed-neighborhood distance samples based on original coordinates
        norm_dist_ss_regional = {"place_type_1": [], "place_type_2": [], "place_type_3": []} # dictionary for fixed-neighborhood distance samples based on normalized coordinates
        reg_dist_ss_global = [] # global single model fixed-neighborhood distance samples on original coordinates
        norm_dist_ss_global = [] # global single model fixed-neighborhood distance samples on normalized coordinates


        reg_k_ss_regional = {"place_type_1": [], "place_type_2": [], "place_type_3": []} # dictionary for K nearest samples based on original coordinates
        norm_k_ss_regional = {"place_type_1": [], "place_type_2": [], "place_type_3": []} # dictionary for K nearest samples based on normalized coordinates
        reg_k_ss_global = [] # global single model k nearest sample on original coordinates
        norm_k_ss_global = [] # global single model k nearest sample on normalized coordinates

        samples = fov_coordinates.Sample.unique() # get all mel (different tissue samples that contains fovs)
        regs = sorted(fov_coordinates.region.unique()) # get all regions (Normal, Interface, Tumor)
        if based_on_distance: # is it bsaed on distance
            if based_on_region: # if we have multiple models
                for i in range(len(regs)): 
                    reg_fov_coordinates = fov_coordinates[fov_coordinates.region == regs[i]] # get all tissue samples that contain given region (e.g., interface)
                    locations = model_loc[regs[i]]
                    distances = model_dist[regs[i]]   
                    model_loc_reg, model_loc_norm = [(locations[0],locations[1])],[(locations[2],locations[3])] # derived model location regular, derived model location on normalized
                    dist_reg, dist_norm = distances[0],distances[1] # derived distance regular, derived distance on normalized
                    for s in samples: 
                        fov = reg_fov_coordinates[reg_fov_coordinates.Sample == s] # a selected fov based on region in a ts (tissue sample)
                        fovs1, fovs2 = self.derived_dist_select_samples(fov, model_loc_reg, model_loc_norm, dist_reg, dist_norm) # get samples
                        reg_dist_ss_regional, norm_dist_ss_regional = self.update_list_dic(reg_dist_ss_regional, norm_dist_ss_regional, fovs1, fovs2, regs[i]) 
                
                reg_d_ss_r = {"place_type_1": [], "place_type_2": [], "place_type_3": []}
                norm_d_ss_r = {"place_type_1": [], "place_type_2": [], "place_type_3": []}
                for key in reg_d_ss_r.keys():
                    reg_d_ss_r[key] = flattern_list(reg_dist_ss_regional[key])
                    norm_d_ss_r[key] = flattern_list(norm_dist_ss_regional[key])

                return reg_d_ss_r, norm_d_ss_r
            else:
                model_loc_reg, model_loc_norm = [(model_loc[0],model_loc[1])],[(model_loc[2],model_loc[3])] # derived model location regular, derived model location on normalized
                dist_reg, dist_norm = model_dist[0], model_dist[1] # derived distance regular, derived distance on normalized
                for s in samples:
                    fov = fov_coordinates[fov_coordinates.Sample == s] # a selected fov based on region in a ts (tissue sample)
                    fovs1, fovs2 = self.derived_dist_select_samples(fov, model_loc_reg, model_loc_norm, dist_reg, dist_norm)
                    reg_dist_ss_global.append(fovs1)
                    norm_dist_ss_global.append(fovs2)
                
                reg_d_ss_g = []
                norm_d_ss_g = []
                reg_d_ss_g = flattern_list(reg_dist_ss_global)
                norm_d_ss_g = flattern_list(norm_dist_ss_global)
                return reg_d_ss_g, norm_d_ss_g
        else:
            if based_on_region: # if we have multiple models
                for i in range(len(regs)):
                    reg_fov_coordinates = fov_coordinates[fov_coordinates.region == regs[i]] # get all tissue samples that contain given region (e.g., interface)
                    locations = model_loc[regs[i]]
                    model_loc_reg, model_loc_norm = [(locations[0],locations[1])],[(locations[2],locations[3])] # derived model location regular, derived model location on normalized
                    for s in samples: 
                        fov = reg_fov_coordinates[reg_fov_coordinates.Sample == s] # a selected fov based on region in a ts (tissue sample)
                        fovs1, fovs2 = self.derived_K_select_samples(fov, model_loc_reg, model_loc_norm,k) # get samples
                        reg_k_ss_regional, norm_k_ss_regional = self.update_list_dic(reg_k_ss_regional, norm_k_ss_regional, fovs1, fovs2, regs[i]) 

                reg_k_ss_r = {"place_type_1": [], "place_type_2": [], "place_type_3": []}
                norm_k_ss_r = {"place_type_1": [], "place_type_2": [], "place_type_3": []}
                for key in reg_k_ss_r.keys():
                    reg_k_ss_r[key] = flattern_list(reg_k_ss_regional[key])
                    norm_k_ss_r[key] = flattern_list(norm_k_ss_regional[key])

                return reg_k_ss_r, norm_k_ss_r
            else:
                model_loc_reg, model_loc_norm = [(model_loc[0],model_loc[1])],[(model_loc[2],model_loc[3])] # derived model location regular, derived model location on normalized
                for s in samples:
                    fov = fov_coordinates[fov_coordinates.Sample == s] # a selected fov based on region in a ts (tissue sample)
                    fovs1, fovs2 = self.derived_K_select_samples(fov, model_loc_reg, model_loc_norm, k)
                    reg_k_ss_global.append(fovs1)
                    norm_k_ss_global.append(fovs2)


                reg_k_ss_g = []
                norm_k_ss_g = []
                reg_k_ss_g = flattern_list(reg_k_ss_global)
                norm_k_ss_g = flattern_list(norm_k_ss_global)
                
                return reg_k_ss_g, norm_k_ss_g

if __name__ == "__main__":
    in_file_fov_locs = './datasets/MelPanel4FOVLocations.csv'
    paths = ["./datasets/place_types_LocationsTrain.csv", "./datasets/place_types_LocationsVal.csv", "./datasets/place_types_LocationsTest.csv"]
    dic = {"Train": [], "Valid": [], "Test": []}
    for key, index in zip(dic.keys(),range(3)):
        sampleSelection = SampleSelection(paths[index], const_model_loc= True, k=8, based_on_distance = False, based_on_reg=False, norm = True)
        reg_samples, norm_samples = sampleSelection.select_samples(sampleSelection.k, sampleSelection.based_on_region, sampleSelection.based_on_distance)
        dic[key] = [reg_samples, norm_samples]
    pickle.dump(dic, open("./datasets/[dataset_name].p", "wb" ))