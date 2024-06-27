import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import pdist, dice
from scipy.cluster import hierarchy
from scipy import stats
import random
import numpy as np
import itertools
from nilearn.maskers import NiftiMasker
from nilearn import datasets, plotting, image, surface
import nibabel as nib
import seaborn as sns
import os, glob, json
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from compute_similarity import compute_similarity
from skimage.metrics import variation_of_information

#TODO add launching of config file if already exists

class FC_Parcellation:
    '''
    The FC_Parcellation class is used to perform the parcellation of a specific roi
    based on the FC profiles of each of its voxels
    
    Attributes
    ----------
    config : dict
    struct_source/data_target : str
        source (i.e., to parcellate) and target (i.e., the one with wich each voxel of the source is correlated) structures
        default is struct_source = 'spinalcord', struct_target = 'brain'
    fc_metric: str
        defines metrics to used to compute functional connectivity (e.g., 'corr' or 'mi') (default = 'corr')
    clusters : dict
        will contain the labeling results for each subject
    data_source/target : dict of array
        contains 4D data of all subjects for the structures of interest
    params_kmeans : dict
        parameters for k-means clustering
        - init: method for initialization (default = 'k-means++')
        - n_init: number of times the algorithm is run with different centroid seeds (default = 256)
        - max_iter: maximum number of iterations of the k-means algorithm for a single run (default = 10000)
    params_spectral : dict
        parameters for spectral clustering
        - n_init: number of times the algorithm is run with different centroid seeds (default = 256)
        - affinity: how to contruct the affinity matrix (common choices: 'nearest_neighbors' or 'rbf') (default = 'nearest_neighbors')
        - assign_labels: trategy for assigning labels in the embedding space (default = 'kmeans')
        - eigen_solver: eigenvalue decomposition strategy to use (default = 'arpack')
        - eigen_tol: stopping criterion for eigendecomposition of the Laplacian matrix (default = 1.0e-5)
    params_agglom : dict
        parameters for agglomerative clustering
        - linkage: distance to use between sets of observations (default = 'average')
        - metric: metric used to compute the linkage (default = 'precomputed')
    '''
    
    def __init__(self, config, struct_source='spinalcord', struct_target='brain', fc_metric='corr', params_kmeans={'init':'k-means++', 'n_init':256, 'max_iter':10000}, params_spectral={'n_init':256, 'affinity': 'nearest_neighbors', 'assign_labels': 'kmeans', 'eigen_solver': 'arpack', 'eigen_tol': 1.0e-5}, params_agglom={'linkage':'average', 'metric':'precomputed'}):
        self.config = config # Load config info
        self.struct_source = struct_source
        self.struct_target = struct_target
        self.fc_metric=fc_metric
        self.clusters = {}
        
        self.init = params_kmeans.get('init')
        self.n_init_kmeans = params_kmeans.get('n_init')
        self.max_iter = params_kmeans.get('max_iter')
        self.linkage = params_agglom.get('linkage')
        self.metric = params_agglom.get('metric')

        self.n_init_spectral = params_spectral.get('n_init')
        self.affinity = params_spectral.get('affinity')
        self.assign_labels = params_spectral.get('assign_labels')
        self.eigen_solver = params_spectral.get('eigen_solver')
        self.eigen_tol = params_spectral.get('eigen_tol')

        # Read mask data
        self.mask_source_path = self.config['main_dir']+self.config['masks']['source']
        self.mask_target_path = self.config['main_dir']+self.config['masks']['target']
        self.mask_source = nib.load(self.mask_source_path).get_data().astype(bool)
        self.mask_target = nib.load(self.mask_target_path).get_data().astype(bool)
        if struct_source == 'spinalcord':
            self.levels_masked = nib.load(self.config['main_dir'] + self.config['spinal_levels']).get_fdata()[self.mask_source] 
            self.levels_order = np.argsort(self.levels_masked.astype(int))
            self.levels_sorted = self.levels_masked[self.levels_order]
        
        # Create folder structure and save config file as json for reference
        path_to_create = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/'
        os.makedirs(os.path.dirname(path_to_create), exist_ok=True)
        for folder in ['fcs', 'fcs_slicewise', 'source', 'target']:
            os.makedirs(os.path.join(path_to_create, folder), exist_ok=True)
        path_config = path_to_create + 'config_' + self.config['output_tag'] + '.json'
        with open(path_config, 'w') as f:
            json.dump(self.config,f)
            
    def compute_voxelwise_fc(self, sub, standardize=True, overwrite=False, njobs=10):
        '''
        To compute functional connectivity between each voxel of mask_source to all voxels of mask_target
        The similarity between these functional connectivity profiles is also calculated
        Can be done using Pearson correlation or Mutual Information
        
        Inputs
        ------------
        sub : str 
            subject on which to run the correlations
        standardize : bool, optional
            if set to True, timeseries are z-scored (default = True)
        overwrite : boolean
            if set to True, labels are overwritten (default = False)
        njobs: int
            number of jobs for parallelization [for MI only] (default = 40)

        Output
        ------------
        fc : array
            one array per subject (n_source_voxels x n_target_voxels), saved as a .npy file
        sim : array
            one array per subject (n_source_voxels x n_source_voxels), saved as a .npy file'''
        
        print(f"\033[1mCOMPUTE VOXELWISE FC\033[0m")
        print(f"\033[37mStandardize = {standardize}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        # Compute FC
        # We can load it from file if it exists
        if not overwrite and os.path.isfile(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '.npy'):
            print(f"... FC already computed")
        
        else: # Otherwise we compute FC    
            print(f"... Loading data")
            data_source = nib.load(self.config['main_dir'] + self.config['smooth_dir'] + 'sub-'+ sub + '/' + self.struct_source + '/sub-' + sub + self.config['file_tag'][self.struct_source]).get_fdata() # Read the data as a matrix
            data_target = nib.load(self.config['main_dir'] + self.config['smooth_dir'] + 'sub-'+ sub + '/' + self.struct_target + '/sub-' + sub + self.config['file_tag'][self.struct_target]).get_fdata() 

            print(f"... Computing FC for all possibilities")
            # Create empty array
            fc = np.zeros((np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_target)))
            data_source_masked = data_source[self.mask_source]
            data_target_masked = data_target[self.mask_target] 
            
            if standardize:
                data_source_masked = stats.zscore(data_source_masked, axis=1).astype(np.float32)
                data_target_masked = stats.zscore(data_target_masked, axis=1).astype(np.float32)

            if self.fc_metric == 'corr':
                print("... Metric: correlation")
                fc = self._corr2_coeff(data_source_masked,data_target_masked)
                print(f"... Fisher transforming correlations")
                # Set values slightly below 1 or above -1 (for use with, e.g., arctanh) [FROM CBP TOOLS]
                fc[fc >= 1] = np.nextafter(np.float32(1.), np.float32(-1))
                fc[fc <= -1] = np.nextafter(np.float32(-1.), np.float32(1))
                fc = fc.astype(np.float32)
                fc = np.arctanh(fc)
            elif self.fc_metric == 'mi':
                print("... Metric: mutual information")
                #for vox_source in tqdm(range(np.shape(data_source_masked)[0])):
                #    fc[vox_source,:] = mutual_info_regression(data_target_masked.T, data_source_masked[vox_source,:].T, n_neighbors=8) 
                #    fc[vox_source,:] = np.max(fc[vox_source,:])
                pool = Pool(njobs)
                vox_source_list = list(range(np.shape(data_source_masked)[0]))
                result_list = []
                for result in tqdm(pool.imap_unordered(partial(self._compute_mi, data_source_masked=data_source_masked, data_target_masked=data_target_masked), vox_source_list), total=len(vox_source_list)):
                    result_list.append(result)
                pool.close()
                pool.join()
                for i, result in enumerate(result_list):
                    fc[i, :] = result
                    fc[i, :] /= np.max(fc[i, :])    
                fc = fc.astype(np.float32)

            # Also compute similarity matrix
            print("... Computing similarity matrix")
            sim = np.corrcoef(fc)
            # Save everything   
            np.save(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '.npy',fc)
            np.save(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '_sim.npy',sim)
        
        print("\n\033[1mDONE\033[0m")

    def compute_mean_fc_sim(self, overwrite=False):
        '''
        To compute mean functional connectivity / similarity across all participants
        
        Inputs
        ------------
        overwrite : boolean
            if set to True, labels are overwritten (default = False)
        
        Output
        ------------
        fc_mean : array
            one array for the mean FC across participants (n_source_voxels x n_target_voxels), saved as a .npy file
        sim_mean : array
            one array for the mean similarity matrix across participants (n_source_voxels x n_source_voxels), saved as a .npy file
        '''
        
        print(f"\033[1mCOMPUTE MEAN ACROSS PARTICIPANTS\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        # Compute mean FC
        # We can load it from file if it exists
        path_mean_fc = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_mean_' + self.fc_metric + '.npy'
        if not overwrite and os.path.isfile(path_mean_fc):
            print(f"... Mean FC already computed")
        else:
            print(f"... Computing mean FC")
            fc_all = np.zeros((len(self.config['list_subjects']),np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_target)))
            for sub_id,sub_name in enumerate(self.config['list_subjects']):
                fc_all[sub_id,:,:] = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub_name + '_' + self.fc_metric + '.npy')            
            fc_mean = np.mean(fc_all, axis=0)
            np.save(path_mean_fc,fc_mean)

        # Compute mean similarity matrix 
        # We can load it from file if it exists
        path_mean_sim = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_mean_' + self.fc_metric + '_sim.npy'
        if not overwrite and os.path.isfile(path_mean_sim):
            print(f"... Mean similarity matrix already computed")
        else:
            print(f"... Computing mean similarity matrix")
            sim_all = np.zeros((len(self.config['list_subjects']),np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_source)))
            for sub_id,sub_name in enumerate(self.config['list_subjects']):
                sim_all[sub_id,:,:] = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub_name + '_' + self.fc_metric + '_sim.npy')            
            sim_mean = np.mean(sim_all, axis=0)
            np.save(path_mean_sim,sim_mean)

        print("\n\033[1mDONE\033[0m")           

    def split_half_validity(self, stability=False, k_selection=False, k_range=None, reps=100, overwrite=False):
        '''  
        ONLY FOR FEATURES = 'SIM' and ALGORITHM = 'AGGLOM' 
        Run split-half validation analyses:
        - correlation between similarity matrix computed from two random halves of the dataset
        - Dice coefficients between adjacency matrix obtained with clustering computed from two random halves of the dataset

        Inputs
        ------------
        stability : boolean
            to compute split-half stability between similarity matrices (default = False)
        k_selection : boolean
            to compute Dice coefficients between split-half clustering (default = False)
        k_range : int, array or range [Only if k_selection == True] 
            number of clusters  
        reps : int
            number of times the dataset is randomly split
        overwrite : boolean
            if set to True, labels are overwritten (default = False)
        
        Output
        ------------
        splithalf_k_selection_df : df
            contains Dice coefficients between split-half clustering for each K and repetition
        stability.nii.gz / npy
            contains average similarity map (in nifti) and similarity maps for each repetition (as npy)
        '''
        
        print(f"\033[1mSPLIT-HALF VALIDATION\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        # Path to create folder structure
        path_source = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/sim/'
        os.makedirs(os.path.join(path_source, 'splithalf_validity'), exist_ok=True)

        if stability:
            path_stability = path_source + 'splithalf_validity/' + self.config['output_tag'] + '_stability'
            if not overwrite and os.path.isfile(path_stability + '.nii.gz'):
                print(f"\033[1mSplit-half maps stability already computed \033[0m")
                stability = False # Set to False as we don't need to redo it
            else:
                print(f"\033[1mSplit-half maps stability will be computed \033[0m")
                stability_maps = np.zeros((reps,np.count_nonzero(self.mask_source))) # Prepare structure
        if k_selection:
            if k_range is None:
                raise(Exception(f'Parameters "k_range" needs to be provided!'))  
            # If only one k value is given, convert to range
            k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range
            path_splithalf_k_selection = path_source + 'splithalf_validity/' + self.config['output_tag'] + '_splithalf_k_selection.pkl'
            # Load file if it exists
            if os.path.isfile(path_splithalf_k_selection): # Take it if it exists
                splithalf_k_selection_df = pd.read_pickle(path_splithalf_k_selection)
                if not overwrite:
                    # Then we check if our ks of interest have already been done
                    k_not_done = [k not in splithalf_k_selection_df['k'].unique() for k in k_range]
                    k_range = [k for k, m in zip(k_range, k_not_done) if m] # Keep only values that have not been done
                    if k_range == []: # if no value left, no analysis to do
                        print(f"\033[1mSplit-half clustering stability already computed for provided K values \033[0m")
                        k_selection = False
                    else:
                        print(f"\033[1mSplit-half clustering stability will be computed for {k_range}\033[0m")
                else:
                    # We overwrite the rows corresponding to k_range
                    splithalf_k_selection_df = splithalf_k_selection_df[~splithalf_k_selection_df['k'].isin(k_range)].reset_index(drop=True)
                    print(f"\033[1mSplit-half clustering stability will be computed for {k_range} \033[0m")
            else: # Create an empty dataframe with the needed columns
                print(f"\033[1mSplit-half clustering stability will be computed for {k_range} \033[0m")
                columns = ["rep", "dice", "ami", "ari", "vi", "k"]
                splithalf_k_selection_df = pd.DataFrame(columns=columns)
    
        # If both stability and k_selection are False, stop further execution
        if not stability and not k_selection:
            print("No analysis to run!")
            return
            
        # Otherwise, for both analyses, we need to load either similarity matrices
        print(f"... Loading similarity matrices")
        sim_all = np.zeros((len(self.config['list_subjects']),np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_source)))
        for sub_id,sub_name in enumerate(self.config['list_subjects']):
            sim_all[sub_id,:,:] = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub_name + '_' + self.fc_metric + '_sim.npy')            

        # Randomly split dataset 
        for rep in range(reps):
            print(f"... Rep: {rep+1}")
            random.sample(range(31), 15)
            half1_idx = random.sample(range(len(self.config['list_subjects'])), len(self.config['list_subjects'])//2)
            half2_idx = np.setdiff1d(np.arange(len(self.config['list_subjects'])), half1_idx)
            # Compute average connectivity matrix for the two halves dataset
            sim_mean_half1 = np.mean(sim_all[half1_idx,:,:],axis=0)
            sim_mean_half2 = np.mean(sim_all[half2_idx,:,:],axis=0)

            if k_selection:
                # Cluster each half
                up_tri_half1 = sim_mean_half1[np.triu_indices(sim_mean_half1.shape[0], k=1)]
                linkage_half1 = hierarchy.linkage(1-up_tri_half1, method=self.linkage)        
                up_tri_half2 = sim_mean_half2[np.triu_indices(sim_mean_half2.shape[0], k=1)]
                linkage_half2 = hierarchy.linkage(1-up_tri_half2, method=self.linkage)    

                # Cut at different K
                for k in k_range:
                    print(f"...... Running clustering stability analysis for K = {k}")
                    kvalues_half1 = hierarchy.cut_tree(linkage_half1, n_clusters=k).flatten().astype(int)
                    kvalues_half2 = hierarchy.cut_tree(linkage_half2, n_clusters=k).flatten().astype(int)    
                    # Compute adjacency metrics and extract Dice coefficients
                    A_half1 = self._cluster_indices_to_adjacency(kvalues_half1)
                    A_half2 = self._cluster_indices_to_adjacency(kvalues_half2)
                    A_half1_tri = A_half1[np.triu_indices(A_half1.shape[0], k=1)]    
                    A_half2_tri = A_half2[np.triu_indices(A_half2.shape[0], k=1)] 
                    splithalf_k_selection_df.loc[len(splithalf_k_selection_df)] = [sub_name,dice(A_half1_tri,A_half2_tri),adjusted_mutual_info_score(kvalues_half1.flatten(),kvalues_half2.flatten()),adjusted_rand_score(kvalues_half1.flatten(),kvalues_half2.flatten()), variation_of_information(kvalues_half1.flatten(),kvalues_half2.flatten())[0], k]
 
            if stability: # Compute correlation with the left-out participant for each source voxel
                print(f"...... Running maps stability analysis")
                for vox in range(np.count_nonzero(self.mask_source)):
                    stability_maps[rep,vox], _ = stats.pearsonr(sim_mean_half1[vox], sim_mean_half2[vox])
       
        if stability:
                # Save as npy
                np.save(path_stability + '.npy',stability_maps)    
                # Compute average stability maps across reps and save as nifti
                average_stability_map = np.mean(stability_maps, axis=0)
                source_mask = NiftiMasker(self.mask_source_path).fit()
                average_stability_map_img = source_mask.inverse_transform(average_stability_map) 
                average_stability_map_img.to_filename(path_stability + '.nii.gz')

        if k_selection:
            splithalf_k_selection_df.to_pickle(path_splithalf_k_selection) 

        print("\n\033[1mDONE\033[0m") 
        
    def similarity_intersub(self, overwrite=False):
        '''  
        ONLY FOR FEATURES = 'SIM' and ALGORITHM = 'AGGLOM' 
        

        XXX TO DO

    
        '''
        
        print(f"\033[1mSTABILITY ANALYSIS\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        # Path to create folder structure
        path_source = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/fcs/'
        path_stability = path_source + self.config['output_tag'] + '_stability'
        if not overwrite and os.path.isfile(path_stability + '.nii.gz'):
            print(f"\033[1mLOO maps stability already computed \033[0m")
        else:
            # Otherwise, we need to load similarity matrices
            print(f"... Loading similarity matrices")
            sim_all = np.zeros((len(self.config['list_subjects']),np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_source)))
            for sub_id,sub_name in enumerate(self.config['list_subjects']):
                sim_all[sub_id,:,:] = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub_name + '_' + self.fc_metric + '_sim.npy')            

            print(f"... Computing stability across participants")
            # Generate all unique combinations of subject indices
            subject_combinations = list(itertools.combinations(range(len(self.config['list_subjects'])), 2))

            # Iterate over unique subject combinations
            stability_maps = []
            for sub_id_1, sub_id_2 in subject_combinations:
                sim_sub_1 = sim_all[sub_id_1,:,:]
                sim_sub_2 = sim_all[sub_id_2,:,:]
                map_combination = np.zeros((np.count_nonzero(self.mask_source),1))
                for vox in range(np.count_nonzero(self.mask_source)):
                    map_combination[vox], _ = stats.pearsonr(sim_sub_1[vox,:], sim_sub_2[vox,:])
                stability_maps.append(map_combination)
                
            # Save as npy
            np.save(path_stability + '.npy',stability_maps)    
            # Compute average stability maps across leave-one-out steps and save as nifti
            average_stability_map = np.mean(stability_maps, axis=0)
            source_mask = NiftiMasker(self.mask_source_path).fit()
            average_stability_map_img = source_mask.inverse_transform(np.squeeze(average_stability_map)) 
            average_stability_map_img.to_filename(path_stability + '.nii.gz')
            
        print("\n\033[1mDONE\033[0m") 
    
    def loo_validity(self, stability=False, k_selection=False, k_range=None, overwrite=False):
        '''  
        ONLY FOR FEATURES = 'SIM' and ALGORITHM = 'AGGLOM' 
        Run leave-one-out validation analyses:
        - correlation between similarity matrix mean (N-1) and LOO-participant
        - Dice coefficients between adjacency matrix obtained with clustering of mean (N-1) and adjacency matrix for LOO-participant

        Inputs
        ------------
        stability : boolean
            to compute LOO-stability between similarity matrices (default = False)
        k_selection : boolean
            to compute Dice coefficients between LOO-clustering (default = False)
        k_range : int, array or range [Only if k_selection == True] 
            number of clusters  
        overwrite : boolean
            if set to True, labels are overwritten (default = False)
        
        Output
        ------------
        loo_k_selection_df : df
            contains Dice coefficients between LOO-clustering for each K and LOO sub
        stability.nii.gz / npy
            contains average similarity map (in nifti) and similarity maps for each LOO sub (as npy)
        '''
        
        print(f"\033[1mLEAVE-ONE-OUT VALIDATION\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        # Path to create folder structure
        path_source = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/sim/'
        os.makedirs(os.path.join(path_source, 'loo_validity'), exist_ok=True)

        if stability:
            path_stability = path_source + 'loo_validity/' + self.config['output_tag'] + '_stability'
            if not overwrite and os.path.isfile(path_stability + '.nii.gz'):
                print(f"\033[1mLOO maps stability already computed \033[0m")
                stability = False # Set to False as we don't need to redo it
            else:
                print(f"\033[1mLOO maps stability will be computed \033[0m")
                stability_maps = np.zeros((len(self.config['list_subjects']),np.count_nonzero(self.mask_source))) # Prepare structure
        if k_selection:
            if k_range is None:
                raise(Exception(f'Parameters "k_range" needs to be provided!'))  
            # If only one k value is given, convert to range
            k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range
            path_loo_k_selection = path_source + 'loo_validity/' + self.config['output_tag'] + '_loo_k_selection.pkl'
            # Load file if it exists
            if os.path.isfile(path_loo_k_selection): # Take it if it exists
                loo_k_selection_df = pd.read_pickle(path_loo_k_selection)
                if not overwrite:
                    # Then we check if our ks of interest have already been done
                    k_not_done = [k not in loo_k_selection_df['k'].unique() for k in k_range]
                    k_range = [k for k, m in zip(k_range, k_not_done) if m] # Keep only values that have not been done
                    if k_range == []: # if no value left, no analysis to do
                        print(f"\033[1mLOO clustering stability already computed for provided K values \033[0m")
                        k_selection = False
                    else:
                        print(f"\033[1mLOO clustering stability will be computed for {k_range}\033[0m")
                else:
                    # We overwrite the rows corresponding to k_range
                    loo_k_selection_df = loo_k_selection_df[~loo_k_selection_df['k'].isin(k_range)].reset_index(drop=True)
                    print(f"\033[1mLOO clustering stability will be computed for {k_range} \033[0m")
            else: # Create an empty dataframe with the needed columns
                print(f"\033[1mLOO clustering stability will be computed for {k_range} \033[0m")
                columns = ["sub", "dice", "ami", "ari", "vi", "k"]
                loo_k_selection_df = pd.DataFrame(columns=columns)
    
        # If both stability and k_selection are False, stop further execution
        if not stability and not k_selection:
            print("No analysis to run!")
            return
            
        # Otherwise, for both analyses, we need to load either similarity matrices
        print(f"... Loading similarity matrices")
        sim_all = np.zeros((len(self.config['list_subjects']),np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_source)))
        for sub_id,sub_name in enumerate(self.config['list_subjects']):
            sim_all[sub_id,:,:] = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub_name + '_' + self.fc_metric + '_sim.npy')            

        # Iterate over each subject
        for sub_id,sub_name in enumerate(self.config['list_subjects']):
            print(f"... LOO participant: {sub_name}")
            # Compute average connectivity matrix for all subjects except the current one
            sim_mean = np.mean(np.delete(sim_all, sub_id, axis=0), axis=0)
            sim_sub = sim_all[sub_id,:,:]

            if k_selection:
                # Cluster mean N-1
                up_tri_mean = sim_mean[np.triu_indices(sim_mean.shape[0], k=1)]
                linkage_mean = hierarchy.linkage(1-up_tri_mean, method=self.linkage)        
                up_tri_sub = sim_sub[np.triu_indices(sim_sub.shape[0], k=1)]
                linkage_sub = hierarchy.linkage(1-up_tri_sub, method=self.linkage)   
                            
                # Cut at different K
                for k in k_range:
                    print(f"...... Running clustering stability analysis for K = {k}")
                    kvalues_mean = hierarchy.cut_tree(linkage_mean, n_clusters=k).flatten().astype(int)
                    kvalues_sub = hierarchy.cut_tree(linkage_sub, n_clusters=k).flatten().astype(int)    
                    # Compute adjacency metrics and extract Dice coefficients
                    A_mean = self._cluster_indices_to_adjacency(kvalues_mean)
                    A_sub = self._cluster_indices_to_adjacency(kvalues_sub)
                    A_mean_tri = A_mean[np.triu_indices(A_mean.shape[0], k=1)]    
                    A_sub_tri = A_sub[np.triu_indices(A_sub.shape[0], k=1)] 
                    loo_k_selection_df.loc[len(loo_k_selection_df)] = [sub_name,dice(A_mean_tri,A_sub_tri),adjusted_mutual_info_score(kvalues_mean.flatten(),kvalues_sub.flatten()),adjusted_rand_score(kvalues_mean.flatten(),kvalues_sub.flatten()), variation_of_information(kvalues_mean.flatten(),kvalues_sub.flatten())[0], k]

            if stability: # Compute correlation with the left-out participant for each source voxel
                print(f"...... Running maps stability analysis")
                for vox in range(np.count_nonzero(self.mask_source)):
                    stability_maps[sub_id,vox], _ = stats.pearsonr(sim_mean[vox], sim_all[sub_id, vox])
        
        if stability:
                # Save as npy
                np.save(path_stability + '.npy',stability_maps)    
                # Compute average stability maps across leave-one-out steps and save as nifti
                average_stability_map = np.mean(stability_maps, axis=0)
                source_mask = NiftiMasker(self.mask_source_path).fit()
                average_stability_map_img = source_mask.inverse_transform(average_stability_map) 
                average_stability_map_img.to_filename(path_stability + '.nii.gz')
        if k_selection:
            loo_k_selection_df.to_pickle(path_loo_k_selection) 
        print("\n\033[1mDONE\033[0m") 
    
    def run_clustering(self, k_range, algorithm, sub=None, take_mean=False, features='fc', save_visplot_sc=True, save_nifti=True, overwrite=False):
        '''  
        Run clustering for a range of k values
        Saving validity metrics using two methods:
        - SSE (not if 'agglom' is used)
        - Silhouette coefficients
        - Davies-Bouldin index 
        - Calinski-Harabasz index

        Inputs
        ------------
        sub : str 
            subject on which to run the correlations
            Note: not considered if take_mean = True
        k_range : int, array or range
            number of clusters  
        algorithm : str
            defines which algorithm to use ('kmeans', 'spectral', or 'agglom')
        take_mean : boolean
            if set to True, clustering is done on the mean values (either FC or sim) across all participants
        features : str
            defines if fc ('fc') or similarity matrix ('sim') (i.e., correlations between the fc profiles) is used as features (default = 'fc')
        save_visplot_sc : boolean
            if set to True, plots are saved to visualize similarity matrix for the spinal cord (default = True)
        save_nifi : boolean
            if set to True, labels are saved as .nii.gz (default = True)
        overwrite : boolean
            if set to True, labels are overwritten (default = False)
        
        Output
        ------------
        dict_clustering_indiv/mean :  dict
            labels corresponding to the clustering for each k (e.g., dict_clustering_indiv['5'] contains labels of a particular subject for k=5)
            Note: there is one dict per subject or for the mean, saved as a .pkl file for easy access
        internal_validity_df : dataframe
            contains the validity metrics for the clustering at the individual level
        '''

        print(f"\033[1mCLUSTERING AT THE INDIVIDUAL LEVEL\033[0m")
        print(f"\033[37mAlgorithm = {algorithm}\033[0m")
        print(f"\033[37mFeatures = {features}\033[0m")
        print(f"\033[37mK value(s) = {k_range}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        if take_mean:
            print(f"\033[1mClustering will be done on the mean across participants!\033[0m")

        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range

        # Path to create folder structure
        path_source = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/' + features + '/'
        os.makedirs(os.path.join(path_source, 'validity'), exist_ok=True)
        path_internal_validity = path_source + 'validity/' + self.config['output_tag'] + '_' + algorithm + ''.join(['_internal_validity_mean.pkl' if take_mean else '_internal_validity.pkl'])
        # Load file with metrics to define K (SSE, silhouette, ...)
        if os.path.isfile(path_internal_validity): # Take it if it exists
            internal_validity_df = pd.read_pickle(path_internal_validity)
            if overwrite: # Overwrite subject if already done and option is set
                internal_validity_df = internal_validity_df[~((internal_validity_df['sub'] == sub) & internal_validity_df['k'].isin(k_range))]

        else: # Create an empty dataframe with the needed columns
            columns = ["sub", "SSE", "silhouette", "davies", "calinski", "k"]
            internal_validity_df = pd.DataFrame(columns=columns)

        fc_not_loaded = True # To avoid loading fc multiple times
        # Check if file already exists
        for k in k_range:
            print(f"K = {k}")
            
            # Create folder structure if needed
            for folder in ['indiv_labels','indiv_labels_relabeled','group_labels','mean_labels']:
                os.makedirs(os.path.join(path_source, 'K'+str(k), folder), exist_ok=True)

            # Check if file already exists
            path_indiv_labels = path_source + 'K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_' + algorithm + '_labels_k' + str(k) if take_mean else path_source + 'K' + str(k) + '/indiv_labels/' + self.config['output_tag'] + '_' + sub + '_' + algorithm + '_labels_k' + str(k)
            
            if not overwrite and os.path.isfile(path_indiv_labels + '.npy'):
                print(f"... Labels already computed")
            
            # Otherwise, we compute them
            else:
                if fc_not_loaded is True:
                    print(f"... Loading FC from file")
    
                    if features == 'fc':
                        if take_mean:
                            data_to_cluster = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_mean_' + self.fc_metric + '.npy')
                        else:
                            data_to_cluster = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '.npy')
                                                
                    elif features == 'sim':
                        print(f"... Loading similarity matrix from file")
                        
                        if take_mean:
                            data_to_cluster = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_mean_' + self.fc_metric + '_sim.npy')
                        else:
                            data_to_cluster = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '_sim.npy')
                        
                        # Reorder for visualization purposes, to see if it follows levels for the spinal cord
                        if self.struct_source == 'spinalcord' and save_visplot_sc is True:
                            data_to_cluster_r = data_to_cluster[self.levels_order,:]
                            data_to_cluster_r = data_to_cluster_r[:,self.levels_order]
                            # Preparing some plots for visualization purposes
                            os.makedirs(os.path.join(path_source, 'visualization'), exist_ok=True)
                            self._plot_sim(sim_r=data_to_cluster_r,save_path=path_source + '/visualization/',sub='mean' if take_mean else sub)
                            self._plot_pca(sim_r=data_to_cluster_r,save_path=path_source + '/visualization/',sub='mean' if take_mean else sub)

                    fc_not_loaded = False # To avoid loading fc multiple times

                if algorithm == 'kmeans':
                    print(f"... Running k-means clustering")
                    # Dict containing k means parameters
                    kmeans_kwargs = {'n_clusters': k, 'init': self.init, 'max_iter': self.max_iter, 'n_init': self.n_init_kmeans}

                    # Compute clustering
                    kmeans_clusters = KMeans(**kmeans_kwargs)
                    kmeans_clusters.fit(data_to_cluster)
                    labels = kmeans_clusters.labels_

                    # Compute validity metrics and add them to dataframe
                    if features == 'fc':
                        internal_validity_df.loc[len(internal_validity_df)] = [sub, kmeans_clusters.inertia_, silhouette_score(data_to_cluster, labels), davies_bouldin_score(data_to_cluster, labels), calinski_harabasz_score(data_to_cluster, labels), k]
                    elif features == 'sim':
                        internal_validity_df.loc[len(internal_validity_df)] = [sub, kmeans_clusters.inertia_, silhouette_score(1-data_to_cluster, labels, metric='precomputed'), davies_bouldin_score(data_to_cluster, labels), calinski_harabasz_score(data_to_cluster, labels), k]
                    
                elif algorithm == 'spectral':
                    print(f"... Running spectral clustering")
                    
                    spectral_kwargs = {'n_clusters': k, 'n_init': self.n_init_spectral, 'affinity': self.affinity,
                        'assign_labels': self.assign_labels, 'eigen_solver': self.eigen_solver, 'eigen_tol': self.eigen_tol} 
                   
                    spectral_clusters = SpectralClustering(**spectral_kwargs)
                    spectral_clusters.fit(data_to_cluster)
                    labels = spectral_clusters.labels_
                    
                    internal_validity_df.loc[len(internal_validity_df)] = [sub, 0, silhouette_score(data_to_cluster, labels), davies_bouldin_score(data_to_cluster, labels), calinski_harabasz_score(data_to_cluster, labels), k]

                elif algorithm == 'agglom':
                    print(f"... Running agglomerative clustering")
                    # Dict containing parameters
                    if features == 'sim':
                        agglom_kwargs = {'n_clusters': k, 'linkage': self.linkage, 'metric': self.metric}

                        agglom_clusters = AgglomerativeClustering(**agglom_kwargs)
                        agglom_clusters.fit(1-data_to_cluster)
                        labels = agglom_clusters.labels_

                        # Compute validity metrics and add them to dataframe
                        # Note that SSE is not relevant for this type of clustering
                        internal_validity_df.loc[len(internal_validity_df)] = [sub, 0, silhouette_score(1-data_to_cluster, labels, metric='precomputed'), davies_bouldin_score(data_to_cluster, labels), calinski_harabasz_score(data_to_cluster, labels), k]
                    else:
                        raise(Exception(f'Algorithm {algorithm} can only be used with FC similarity.'))    
                else:
                    raise(Exception(f'Algorithm {algorithm} is not a valid option.'))
            
                np.save(path_indiv_labels + '.npy', labels.astype(int))
            
                if save_nifti:
                    seed = NiftiMasker(self.mask_source_path).fit()
                    labels_img = seed.inverse_transform(labels.astype(int)+1) # +1 because labels start from 0 
                    labels_img.to_filename(path_indiv_labels + '.nii.gz')

        internal_validity_df.to_pickle(path_internal_validity) 

        print("\n")
               
    def group_clustering(self, k_range, indiv_algorithm, features='fc', linkage='ward', overwrite=False):
        '''  
        Perform group-level clustering using the individual labels
        BASED ON CBP toolbox

        Inputs
        ------------
        k_range : int, array or range
            number of clusters  
        indiv_algorithm : str
            algorithm that was used at the subject level
        linkage : str
            define type of linkage to use for hierarchical clustering (default = 'ward')
        features : str
            defines if fc ('fc') or similarity matrix ('sim') (i.e., correlations between the fc profiles) is used as features (default = 'fc')
        overwrite : boolean
            if set to True, labels are overwritten (default = False)
       
        Output
        ------------
        Saving relabeled labels for each individual participant (as .npy)
        Saving group labels (as .npy and .nii.gz)
            (either using the mode of the relabeled participant-wise clustering, or using the hierarchical clustering result)

        '''
        print(f"\033[1mCLUSTERING AT THE GROUP LEVEL\033[0m")
        print(f"\033[37mK value(s) = {k_range}\033[0m")
        print(f"\033[37mFeatures = {features}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
       
        if overwrite == False:
            print("\033[38;5;208mWARNING: THESE RESULTS CHANGE IF GROUP CHANGES, MAKE SURE YOU ARE USING THE SAME PARTICIPANTS\033[0m\n")
        
        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range

        path_source = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/' + features + '/'
        path_group_validity = path_source + 'validity/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_validity.pkl'
        path_cophenetic_correlation = path_source + 'validity/' + self.config['output_tag'] + '_' + indiv_algorithm + '_cophenetic_correlation.pkl'
        
        # Load file with metrics to similarity indiv vs group labels
        if not overwrite and os.path.isfile(path_group_validity):
            group_validity_df = pd.read_pickle(path_group_validity) 
        else: # Create an empty dataframe with the needed columns
            columns = ["sub", "ami_mode", "ari_mode", "ami_agglom", "ari_agglom", "k"]
            group_validity_df = pd.DataFrame(columns=columns)

        # Load file with cophenetic correlations
        if not overwrite and os.path.isfile(path_cophenetic_correlation):
            cophenetic_correlation_df = pd.read_pickle(path_cophenetic_correlation) 
        else: # Create an empty dataframe with the needed columns
            columns = ["corr", "k"]
            cophenetic_correlation_df = pd.DataFrame(columns=columns)
        
        # Loop through K values
        for k in k_range:
            print(f"K = {k}")
            
            # We assume that if group_labels exist, the rest is done too
            group_labels_mode_path = path_source + 'K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_mode_k' + str(k) 
            group_labels_agglom_path = path_source + 'K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_agglom_k' + str(k)
            if not overwrite and os.path.isfile(group_labels_mode_path + '.npy'):
                print(f"... Group labeling already done!")
            else:
                print(f"... Computing hierarchical clustering and relabeling")
                # Load subjects data
                # Prepare empty structure
                nvox_source = np.count_nonzero(self.mask_source)
                indiv_labels_all = np.zeros((len(self.config['list_subjects']),nvox_source))
                
                print(f"...... Loading all subjects")                
                for sub_id,sub in enumerate(self.config['list_subjects']):
                    indiv_labels_path =  path_source + '/K' + str(k) + '/indiv_labels/' + self.config['output_tag'] + '_' + sub + '_' + indiv_algorithm + '_labels_k' + str(k) + '.npy'    
                    if os.path.isfile(indiv_labels_path):
                        indiv_labels_all[sub_id,:] = np.load(indiv_labels_path)    
                    else:      
                        raise(Exception(f'Subject {sub} is missing for algorithm {indiv_algorithm} and K={k}.'))
                    
                # Hierarchical clustering on all labels
                print(f"...... Clustering on all labels")
                x = indiv_labels_all.T
                y = pdist(x, metric='hamming')
                z = hierarchy.linkage(y, method=linkage, metric='hamming')
                
                # Group labels from the agglomerative clustering
                group_labels = hierarchy.cut_tree(z, n_clusters=len(np.unique(x)))
                group_labels = np.squeeze(group_labels)  # (N, 1) to (N,)

                print(f"...... Computing cophenetic correlation")
                # Measure of how well the distances are preserved and add to dataframe
                cophenetic_correlation, *_ = hierarchy.cophenet(z, y)
                cophenetic_correlation_df.loc[len(cophenetic_correlation_df)] = [cophenetic_correlation, k]
                
                # Use the hierarchical clustering as a reference to relabel individual
                # participant clustering results
                indiv_labels_relabeled = np.empty((0, indiv_labels_all.shape[1]), int)

                print(f"...... Relabeling")
                # iterate over individual participant labels (rows)
                for label in indiv_labels_all:
                    x, acc = self._relabel(reference=group_labels, x=label)
                    indiv_labels_relabeled = np.vstack([indiv_labels_relabeled , x])
                
                mode, _ = stats.mode(indiv_labels_relabeled, axis=0, keepdims=False)
                # Set group labels to mode for mapping
                group_labels_mode = np.squeeze(mode)
           
                print(f"...... Computing validity")
                for sub_id, sub in enumerate(self.config['list_subjects']): 
                    ami_mode = adjusted_mutual_info_score(labels_true=group_labels_mode,labels_pred=indiv_labels_relabeled[sub_id,:])
                    ari_mode = adjusted_rand_score(labels_true=group_labels_mode,labels_pred=indiv_labels_relabeled[sub_id,:])
                    ami_agglom = adjusted_mutual_info_score(labels_true=group_labels,labels_pred=indiv_labels_relabeled[sub_id,:])
                    ari_agglom = adjusted_rand_score(labels_true=group_labels,labels_pred=indiv_labels_relabeled[sub_id,:])
                    group_validity_df.loc[len(group_validity_df)] = [sub, ami_mode, ari_mode, ami_agglom, ari_agglom, k]

                # Arrays and df
                path_relabeled = path_source + 'K' + str(k) + '/indiv_labels_relabeled/' + self.config['output_tag'] + '_' + indiv_algorithm + '_labels_relabeled_k' + str(k) + '.npy'
                np.save(path_relabeled,indiv_labels_relabeled.astype(int))
                cophenetic_correlation_df.to_pickle(path_cophenetic_correlation)
                group_validity_df.to_pickle(path_group_validity)

                # Maps
                np.save(group_labels_agglom_path + '.npy', group_labels.astype(int))
                np.save(group_labels_mode_path + '.npy', group_labels_mode.astype(int))
                seed = NiftiMasker(self.mask_source_path).fit()
                labels_img = seed.inverse_transform(group_labels+1) # +1 because labels start from 0 
                labels_img.to_filename(group_labels_agglom_path + '.nii.gz')
                labels_img = seed.inverse_transform(group_labels_mode+1) # +1 because labels start from 0 
                labels_img.to_filename(group_labels_mode_path + '.nii.gz')

        print("\n\033[1mDONE\033[0m")
    
    def subject_variability(self, k_range, overwrite=False):
        '''  
        ONLY FOR FEATURES = 'SIM' and ALGORITHM = 'AGGLOM' 
        Relabel individual labels to match mean ones

        Inputs
        ------------
        k_range : int, array or range
            number of clusters  
        overwrite : boolean
            if set to True, labels are overwritten (default = False)
       
        Output
        ------------
        Saving relabeled labels for all participants (as .npy)
        
        '''
        print(f"\033[1mRELABELING\033[0m")
        print(f"\033[37mK value(s) = {k_range}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range

        # Loop through K values
        for k in k_range:
            print(f"K = {k}")        
            path_source = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/sim/K' + str(k) + '/indiv_labels/' 
            path_ref = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/sim/K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_agglom_labels_k' + str(k) + '.npy'
            # Load file with metrics to similarity indiv vs group labels
            if not overwrite and os.path.isfile(path_source + self.config['output_tag'] + '_all_agglom_labels_k' + str(k) + '_relabeled.npy'):
                print(f"... Relabeling already computed")
            else: # Create an empty dataframe with the needed columns
                ref_labels = np.load(path_ref)
                relabeled_labels = np.zeros((len(self.config['list_subjects']),ref_labels.shape[0]))
                seed = NiftiMasker(self.mask_source_path).fit()
                for sub_id, sub in enumerate(self.config['list_subjects']): 
                    print(f"... Relabeling sub-" + sub)
                    sub_labels = np.load(path_source + self.config['output_tag'] + '_' + sub + '_agglom_labels_k' + str(k) + '.npy')
                    # Relabel the target labels to match the reference labels as closely as possible
                    relabeled_labels[sub_id,:] = self._relabel_labels(ref_labels, sub_labels)
                # Save
                np.save(path_source + self.config['output_tag'] + '_all_agglom_labels_k' + str(k) + '_relabeled.npy', relabeled_labels.astype(int))
                print(f"... Compute distribution for each K")
                k_maps = np.zeros((k,ref_labels.shape[0]))
                for k_i in range(0,k):
                    for sub_id,sub in enumerate(self.config['list_subjects']): 
                        k_maps[k_i,:] = k_maps[k_i,:] + (relabeled_labels[sub_id,:]==k_i).astype(int)
                    labels_img = seed.inverse_transform(k_maps[k_i,:]) 
                    labels_img.to_filename(path_source + self.config['output_tag'] + '_all_agglom_labels_k' + str(k) + '_' + str(k_i+1) + '_relabeled.nii.gz')
                
                # Count how many voxels of each cluster each participant has
                clusters_val_per_sub = np.sum(relabeled_labels.astype(int) == np.arange(7)[:, None, None], axis=2)
                # Check how many participants have a corresponding cluster
                participants_per_cluster = np.sum(clusters_val_per_sub > 0, axis=1)
                # Save as txt file
                np.savetxt(path_source + self.config['output_tag'] + '_all_agglom_labels_k' + str(k) + '_' + str(k_i+1) + '_relabeled.txt',participants_per_cluster,fmt='%d',delimiter=',')

        print("\n\033[1mDONE\033[0m")
        
    def prepare_target_maps(self, label_type, k_range, indiv_algorithm, features='fc', overwrite=False):
        '''  
        To obtain images of the connectivity profiles assigned to each label
        (i.e., mean over the connectivity profiles of the voxel of this K)

        Inputs
        ------------
        label_type : str
            defines the type of labels to use to define connectivity patterns (target)
            'indiv': relabeled labels (i.e., specific to each participant)
            'group_mode': group labels (mode) (i.e., same for all participants)
            'group_agglom': group labels (agglomerative) (i.e., same for all participants)    
            'mean': labels obtained from the mean FC or similarity matrix   
        k_range : int, array or range
            number of clusters  
        indiv_algorithm : str
            algorithm that was used at the participant level
        features : str
            defines if fc ('fc') or similarity matrix ('sim') (i.e., correlations between the fc profiles) is used as features (default = 'fc')
        overwrite : boolean
            if set to True, maps are overwritten (default = False)
        
        Outputs
        ------------
        target_maps : array
            array containing the brain maps for each label and participant (nb_participants x K x n_vox_target)
        K nifti images
            one image for each mean connectivity profile (i.e., one per K)
            (e.g., "brain_pattern_K4_1.nii.gz" for the first cluster out of 4 total clusters)
            
        '''

        print(f"\033[1mPREPARE TARGET MAPS\033[0m")
        print(f"\033[37mType of source labels = {label_type}\033[0m")
        print(f"\033[37mK value(s) = {k_range}\033[0m")
        print(f"\033[37mFeatures = {features}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")

        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range
        
        # Path to create folder structure
        path_target = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/target/' + features + '/'
                           
        # Loop through K values
        for k in k_range:
            print(f"K = {k}")    

            target_npy_path = path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/' + self.config['output_tag'] + '_' + indiv_algorithm + '_targetmaps_k' + str(k) + '.npy' 
            
            if not overwrite and os.path.isfile(target_npy_path):
                print(f"... Target maps already computed")
            else:
                print(f"... Computing target maps")
                # Create folder structure if needed
                for folder in ['maps_indiv','maps_mean']:
                    os.makedirs(os.path.join(path_target, 'K'+str(k), label_type + '_labels', folder), exist_ok=True)
                
                # Initialize array to save target data
                target_maps = np.zeros((len(self.config['list_subjects']),k,np.count_nonzero(self.mask_target)))
                for sub_id,sub in enumerate(self.config['list_subjects']):
                    print(f"...... Subject {sub}")
                    
                    # Load labels
                    if label_type == 'indiv':
                        path_labels = self.config['main_dir'] + self.config['output_dir'] +  '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/source/' + features + '/K' + str(k) + '/indiv_labels_relabeled/' + self.config['output_tag'] + '_' + indiv_algorithm + '_labels_relabeled_k' + str(k) + '.npy'
                        labels = np.load(path_labels)
                        labels =  np.squeeze(labels[sub_id,:].T)
                    elif label_type == 'group_mode':
                        path_labels = self.config['main_dir'] + self.config['output_dir'] +  '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/source/' + features + '/K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_mode_k' + str(k) + '.npy'
                        labels = np.load(path_labels)
                    elif label_type == 'group_agglom':
                        path_labels = self.config['main_dir'] + self.config['output_dir'] +  '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/source/' + features + '/K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_agglom_k' + str(k) + '.npy'
                        labels = np.load(path_labels)
                    elif label_type == 'mean':
                        path_labels =self.config['main_dir'] + self.config['output_dir'] +  '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/source/' + features + '/K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_' + indiv_algorithm + '_labels_k' + str(k) + '.npy'
                        labels = np.load(path_labels)
                    else:
                        raise(Exception(f'Label type {label_type} is not a valid option.'))

                    # Load FC
                    fc = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '.npy')
                    
                    # Compute mean map per label
                    for label in range(0,k):
                        target_maps[sub_id,label,:] = np.mean(fc[np.where(labels==label),:],axis=1)
                        
                    # Save indiv maps as nifti files
                    target_mask = NiftiMasker(self.mask_target_path).fit()
                    
                    for label in np.unique(labels):
                        target_map_img = target_mask.inverse_transform(target_maps[sub_id,label,:])
                        path_target_map_img = path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/maps_indiv' + '/' +  self.config['output_tag'] + '_' + sub + '_' + indiv_algorithm + '_' + label_type + '_labels_targetmap_K' + str(k) + '_' + str(label+1) + '.nii.gz'
                        target_map_img.to_filename(path_target_map_img)

                # Save mean maps as nifti files
                for label in np.unique(labels):      
                    target_map_mean_img = target_mask.inverse_transform(np.mean(target_maps[:,label,:],axis=0))
                    path_target_map_mean_img = path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/maps_mean' + '/' +  self.config['output_tag'] + '_mean_' + indiv_algorithm + '_' + label_type + '_labels_targetmap_K' + str(k) + '_' + str(label+1) + '.nii.gz'
                    target_map_mean_img.to_filename(path_target_map_mean_img)

                # Save array
                np.save(target_npy_path, target_maps)
            
        print("\033[1mDONE\033[0m\n")
        
    def stats_target_maps(self, label_type, k_range, indiv_algorithm, features='fc', overwrite=False):
        '''  
        To conduct statistical analyses of the connectivity profiles assigned to each label
        (i.e., mean over the connectivity profiles of the voxel of this K)

        Inputs
        ------------
        label_type : str
            defines the type of labels to use to define connectivity patterns (target)
            'indiv': relabeled labels (i.e., specific to each participant)
            'group_mode': group labels (mode) (i.e., same for all participants)
            'group_agglom': group labels (agglomerative) (i.e., same for all participants)  
            'mean': labels obtained from the mean FC or similarity matrix    
        k_range : int, array or range
            number of clusters  
        indiv_algorithm : str
            algorithm that was used at the participant level
        features : str
            defines if fc ('fc') or similarity matrix ('sim') (i.e., correlations between the fc profiles) is used as features (default = 'fc')
        overwrite : boolean
            if set to True, maps are overwritten (default = False)
        
        Outputs
        ------------
        stats folder
            folder containing the results of the one-sample T-test conducted using randomise
            
        '''

        print(f"\033[1mRUN STATISTICAL ANALYSIS\033[0m")
        print(f"\033[37mType of source labels = {label_type}\033[0m")
        print(f"\033[37mK value(s) = {k_range}\033[0m")
        print(f"\033[37mFeatures = {features}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")

        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range
        
        # Path to target maps
        path_target = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/target/' + features + '/'

        # Loop through K values
        for k in k_range:
            print(f"K = {k}")    

            path_stats = path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/stats/' + indiv_algorithm 
            
            if not overwrite and os.path.isdir(path_stats):
                print(f"... Statistical analysis already done")
            else:
                # Create folder structure if needed
                os.makedirs(path_stats, exist_ok=True)
                
                for k_ind in range(0,k):
                    # Check if 4D file exists
                    path_target_maps_4d = path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/maps_indiv' + '/' +  self.config['output_tag'] + '_all_' + indiv_algorithm + '_' + label_type + '_labels_targetmap_K' + str(k) + '_' + str(k_ind+1) + '.nii.gz'
                    if not os.path.isfile(path_target_maps_4d):
                        print(f"... Merging target files")
                        path_target_maps_all = path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/maps_indiv' + '/' +  self.config['output_tag'] + '_*_' + indiv_algorithm + '_' + label_type + '_labels_targetmap_K' + str(k) + '_' + str(k_ind+1) + '.nii.gz'
                        run_merge = 'fslmerge -t ' + path_target_maps_4d + ' ' + path_target_maps_all 
                        os.system(run_merge)

                    print(f"... Running statistical analysis")
                    run_randomise = 'randomise -i ' + path_target_maps_4d + ' -m ' + self.mask_target_path + ' -o ' + path_stats + '/K' + str(k) + '_' + str(k_ind+1) + ' -1'
                    os.system(run_randomise)

        print("\033[1mDONE\033[0m\n")

    def winner_takes_all(self, label_type, k, indiv_algorithm, input_type='stats', features='fc', order=None, apply_threshold=None, cluster_threshold=100, overwrite=False):
        '''  
        To generate winner-takes-all maps using the t-statistics obtained for each K

        Inputs
        ------------
        label_type : str
            defines the type of labels to use to define connectivity patterns (target)
            'indiv': relabeled labels (i.e., specific to each participant)
            'group_mode': group labels (mode) (i.e., same for all participants)
            'group_agglom': group labels (agglomerative) (i.e., same for all participants)       
            'mean': labels obtained from the mean FC or similarity matrix  
        k : int, array or range
            number of clusters   
        indiv_algorithm : str
            algorithm that was used at the participant level
        input_type : str
            defines whether to use statistical t-maps ('stats') or correlation maps ('corr') as inputs (default = 'corr')
        features : str
            defines if fc ('fc') or similarity matrix ('sim') (i.e., correlations between the fc profiles) is used as features (default = 'fc')
        order : arr or str
            defines which number is going to be associated with each K value (default = None)
            this is useful as K values do not correspond to segments
            Two options:
                - array containing one value par K, e.g. if K = 7: [1 3 2 7 5 6 4] (if None, we use the sequential order)
                - 'from_file' to indicate that order should be loaded from existing txt file
        apply_threshold : float
            to apply a threshold value on the input t-stats images, a cluster thresholding will also be applied (default = None)
        cluster_threshold : int
            to define the value for cluster thresholding (only if apply_threshold is not None) (default = 100) 
        overwrite : boolean
            if set to True, maps are overwritten (default = False)
        
        Outputs
        ------------
        KX_wta_input_type.nii.gz
            WTA map for a particular K
            
        '''

        print(f"\033[1mRUN WINNER-TAKES-ALL ANALYSIS\033[0m")
        print(f"\033[37mType of source labels = {label_type}\033[0m")
        print(f"\033[37mK value = {k}\033[0m")
        print(f"\033[37mFeatures = {features}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")


        # Path to stats
        path_target = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/target/' + features + '/'
        path_stats = path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/stats/' + indiv_algorithm 
        output_file = path_stats + '/K' + str(k) + '_wta_' + input_type + '.nii.gz'
        order_file = path_stats + '/K' + str(k) + '_wta_' + input_type + '_order.txt'

        if not overwrite and os.path.exists(output_file):
            print(f"... WTA analysis already done")
        else:
            # Define order
            if order is None:
                order = range(1,k+1)
            elif order == 'from_file':
                if os.path.exists(order_file):
                    order = np.loadtxt(order_file).astype(int)[1:]
                else:
                    raise(Exception(f'Order file could not be found.')) 
            else:
                order = order

            if len(order) != k:
                raise(Exception(f'The length of the order information ({len(order)}) is different from K {k}.'))  

            for k_ind in range(0,k):
                print('K' + str(k_ind+1) + ' will have a value of ' + str(order[k_ind]))

            order = np.concatenate(([0], order)) # Add a 0 to the beginning of the re-ordering info (for the background)
            
            # Select mask:
            masker = NiftiMasker(self.mask_target_path,smoothing_fwhm=[0,0,0], t_r=1.55, low_pass=None, high_pass=None)

            # Find the t max value for each voxel (group level) 
            maps_file = []; maps_data = []
            for k_ind in range(0,k):
                if input_type == 'stats':
                    maps_file.append(glob.glob(path_stats + '/K' + str(k) + '_' + str(k_ind+1) + '_tstat1.nii.gz')[0]) # select individual maps   
                elif input_type == 'corr':
                    maps_file.append(glob.glob(path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/maps_mean' + '/' +  self.config['output_tag'] + '_mean_' + indiv_algorithm + '_' + label_type + '_labels_targetmap_K' + str(k) + '_' + str(k_ind+1) + '.nii.gz')[0])
                else:
                    raise(Exception(f'Input type should be "stats" or "corr".'))      
                if apply_threshold is not None:
                    maps_thr = image.threshold_img(maps_file[k_ind], threshold=apply_threshold, cluster_threshold=cluster_threshold, mask_img=self.mask_target_path)
                    #maps_thr.to_filename(maps_file[k_ind].split(".")[0] +"_thr_t"+str(apply_threshold)+".nii.gz") # create temporary 3D files
                    maps_data.append(masker.fit_transform(maps_thr)) # extract the data in a single array

                elif apply_threshold is None:
                    maps_data.append(masker.fit_transform(maps_file[k_ind])) # extract the data in a single array

            data = np.squeeze(np.array(maps_data))
            
            max_level_indices = []
                    
            # Loop through each voxel
            for i in range(0,data.shape[1]):
        
                i_values = data[:,i]  # Get the voxel values

                max_level_index = np.argmax(i_values)  # Find the level that have the max value for this column
                if i_values[max_level_index] == 0 :
                    max_level_index =-1 # if the max value is 0 put -1 to the index
                max_level_indices.append(order[max_level_index+1])  
            # Save the output as an image
            seed_to_voxel_img = masker.inverse_transform(np.array(max_level_indices).T)
            seed_to_voxel_img.to_filename(output_file) # create temporary 3D files

            # Save the order as well
            np.savetxt(order_file,order,fmt='%d',delimiter=',')


        print("\033[1mDONE\033[0m\n")

    def winner_takes_all_fc_mean(self, k, indiv_algorithm, features='fc', save_fc_profiles=True, order=None, overwrite=False):
        '''  
        To generate winner-takes-all maps using the mean FC

        Inputs
        ------------
        k : int, array or range
            number of clusters   
        indiv_algorithm : str
            algorithm that was used at the participant level
        features : str
            defines if fc ('fc') or similarity matrix ('sim') (i.e., correlations between the fc profiles) is used as features (default = 'fc')
        save_fc_profiles : boolean
            if set to True, the mean fc profiles for each cluster are saved as nifti files (default = True)
        order : arr or str
            defines which number is going to be associated with each K value (default = None)
            this is useful as K values do not correspond to segments
            Two options:
                - array containing one value par K, e.g. if K = 7: [1 3 2 7 5 6 4] (if None, we use the sequential order)
                - 'from_file' to indicate that order should be loaded from existing txt file
        overwrite : boolean
            if set to True, maps are overwritten (default = False)
        
        Outputs
        ------------
        KX_wta.nii.gz
            WTA map for a particular K
            
        '''

        print(f"\033[1mRUN WINNER-TAKES-ALL ANALYSIS\033[0m")
        print(f"\033[37mK value = {k}\033[0m")
        print(f"\033[37mFeatures = {features}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")

        # Path to stats
        main_path = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag']
        path_mean_fc = main_path + '/fcs/' + self.config['output_tag'] + '_mean_' + self.fc_metric + '.npy'
        path_labels = main_path + '/source/' + features + '/K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_' + indiv_algorithm + '_labels_k' + str(k) + '.npy'
 
        path_wta = main_path + '/target/' + features + '/K' + str(k) + '/wta_fc_mean/'
        output_file = path_wta + '/K' + str(k) + '_wta.nii.gz'
        order_file = path_wta + '/K' + str(k) + '_wta_order.txt'

        if not overwrite and os.path.exists(output_file):
            print(f"... WTA analysis already done")
        else:
            # Create folder structure if needed
            os.makedirs(path_wta, exist_ok=True)
                
            # Define order
            if order is None:
                order = range(1,k+1)
            elif order == 'from_file':
                if os.path.exists(order_file):
                    order = np.loadtxt(order_file).astype(int)[1:]
                else:
                    raise(Exception(f'Order file could not be found.')) 
            else:
                order = order
                
            if len(order) != k:
                raise(Exception(f'The length of the order information ({len(order)}) is different from K {k}.'))  

            for k_ind in range(0,k):
                print('K' + str(k_ind+1) + ' will have a value of ' + str(order[k_ind]))

            order = np.concatenate(([0], order)) # Add a 0 to the beginning of the re-ordering info (for the background)
            # Load mean FC data and labels
            mean_fc = np.load(path_mean_fc)
            labels = np.load(path_labels)

            # Compute average FC profile for each cluster
            mask_labels = (labels == np.arange(0, k)[:, np.newaxis])
            average_fc_profiles = np.array([np.mean(mean_fc[mask_labels[i]], axis=0) for i in range(k)])           

            # For data saving
            target = NiftiMasker(self.mask_target_path).fit() 
            
            # Save FC profiles?
            if save_fc_profiles:
                for k_ind in range(0,k):
                    fc_profile_img = target.inverse_transform(average_fc_profiles[k_ind,:])
                    fc_profile_img.to_filename(path_wta + '/K' + str(k) + '_' + str(order[k_ind]) + '_fc_profile.nii.gz') 
                    
            # Compute WTA
            max_level_indices = []
                    
            # Loop through each voxel
            for i in range(0,average_fc_profiles.shape[1]):
                i_values = average_fc_profiles[:,i]  # Get the voxel values
                max_level_index = np.argmax(i_values)  # Find the level that have the max value for this column
                if i_values[max_level_index] == 0 :
                    max_level_index =-1 # if the max value is 0 put -1 to the index
                max_level_indices.append(order[max_level_index+1]) 
            
            # Save the output as an image
            # If we take the mean, we need to save the nifti files, as no group clustering will be conducted
            wta_img = target.inverse_transform(np.array(max_level_indices).T)
            wta_img.to_filename(output_file) # create temporary 3D files

            # Save the order as well
            np.savetxt(order_file,order,fmt='%d',delimiter=',')


        print("\033[1mDONE\033[0m\n")

    def plot_brain_map(self, k, indiv_algorithm, showing, group_type=None, input_type='stats', wta_fc_mean=False, colormap=plt.cm.rainbow, features='fc', label_type=None, save_figure=False):
        ''' Plot brain maps on inflated brain
        
        Inputs
        ------------  
        k : int
            number of clusters  
        indiv_algorithm : str
            algorithm that was used at the participant level
        colormap : cmap
            colormap to use, will be discretized (default = plt.cm.rainbow)
        showing : str
            defines whether source or target data should be plotted
        group_type : str [Only for source]
            defines the type of labels to display for source
            'mode': group labels (mode) 
            'agglom': group labels (agglomerative)    
            'mean': labels obtained from the mean FC or similarity matrix  
        input_type : str [Only for target]
            defines whether to plot WTA maps based on statistical t-maps ('stats') or correlation maps ('corr') as inputs (default = 'corr')
        wta_fc_mean : boolean [Only for target]
            set to True to use WTA results coming from mean FC matrix (default = False)
        features : str
            defines if fc ('fc') or similarity matrix ('sim') (i.e., correlations between the fc profiles) is used as features (default = 'fc')
        label_type : str [Only for target]
            defines the type of labels to use to define connectivity patterns (target)
            'indiv': relabeled labels (i.e., specific to each participant)
            'group_mode': group labels (mode) (i.e., same for all participants)
            'group_agglom': group labels (agglomerative) (i.e., same for all participants)       
            'mean': labels obtained from the mean FC or similarity matrix  
        save_figure : boolean
            Set to True to save figure (default = False)

        Outputs
        ------------
        png of the brain maps
            
        '''

        print(f"\033[1mRUN PLOTTING BRAIN MAPS\033[0m")
        print(f"\033[37mK value = {k}\033[0m")
        print(f"\033[37mShowing = {showing}\033[0m")
        print(f"\033[37mFeatures = {features}\033[0m")
        print(f"\033[37mSave figure = {save_figure}\033[0m")

        if showing == 'source':
            if group_type is None:
                raise(Exception(f'The parameter group_type is missing!')) 
            elif group_type == 'agglom' or group_type == 'mode':
                path_data = self.config['main_dir'] + self.config['output_dir'] + self.fc_metric + '/' + self.config['output_tag'] + '/source/' + features + '/K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_' + group_type + '_k' + str(k)
            elif group_type == 'mean':
                path_data = self.config['main_dir'] + self.config['output_dir'] + self.fc_metric + '/' + self.config['output_tag'] + '/source/' + features + '/K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_' + indiv_algorithm + '_labels_k' + str(k)
            print("Source labels are not re-ordered!")
        elif showing == 'target':
            if label_type is None and not wta_fc_mean:
                raise(Exception(f'When plotting target, you need to define which labels you want to use!')) 
            if wta_fc_mean:
                path_data = self.config['main_dir'] + self.config['output_dir'] + self.fc_metric + '/' + self.config['output_tag'] + '/target/' + features + '/K' + str(k) + '/wta_fc_mean/K' + str(k) + '_wta'     
            else:
                path_data = self.config['main_dir'] + self.config['output_dir'] + self.fc_metric + '/' + self.config['output_tag'] + '/target/' + features + '/K' + str(k) + '/' + label_type + '_labels' + '/stats/' + indiv_algorithm + '/K' + str(k) + '_wta_' + input_type 
        else:
            raise(Exception(f'The parameter "showing" should be "target" or "source".')) 
        
        if ~hasattr(colormap, 'colors'): # Create a discretized colormap if is not already discrete
            discretized_colormap = ListedColormap(colormap(np.linspace(0, 1, k+1)))
        elif isinstance(colormap.colors, list): # Test if colormap is discrete 
            discretized_colormap = colormap

        img_to_show = path_data + '.nii.gz'
        for hemi in ['left','right']:
            img_surf = surface.vol_to_surf(img_to_show, self.config['main_dir'] + self.config['brain_surfaces'] + 'rh.pial', radius=0,interpolation='nearest', kind='auto', n_samples=10, mask_img=None, depth=None) if hemi == 'right' else surface.vol_to_surf(img_to_show, self.config['main_dir'] + self.config['brain_surfaces'] + 'lh.pial', radius=0,interpolation='nearest', kind='auto', n_samples=10, mask_img=None, depth=None)
           
            plot = plotting.plot_surf_roi(self.config['main_dir']+self.config['brain_surfaces']+'rh.inflated' if hemi =='right' else self.config['main_dir']+self.config['brain_surfaces']+'lh.inflated', roi_map=img_surf,
                       cmap=discretized_colormap, colorbar=True,
                       hemi=hemi, view='lateral', vmin=1,vmax=k,
                       bg_map=self.config['main_dir']+self.config['brain_surfaces']+'rh.sulc' if hemi=='right' else self.config['main_dir']+self.config['brain_surfaces']+'rh.sulc', #bg_on_data=True,
                       darkness=0.7)
            # If option is set, save results as a png
            if save_figure == True:
                if wta_fc_mean:
                    plot_path = path_data + '_' + hemi + '.png'
                else:
                    plot_path = path_data + '_' + hemi + '_' + input_type + '.png'
                plot.savefig(plot_path)
                    
    def plot_spinal_map(self, k, indiv_algorithm, showing, colormap=plt.cm.rainbow, input_type='corr', wta_fc_mean=False, group_type=None, features='fc', label_type=None, order=None, slice_y=None, show_spinal_levels=True, save_figure=False):
        ''' Plot spinal maps on PAM50 template (coronal views)
        
        Inputs
        ------------  
        k : int
            number of clusters  
        indiv_algorithm : str
            algorithm that was used at the participant level
        showing : str
            defines whether source or target data should be plotted
        input_type : str [Only for target]
            defines whether to plot WTA maps based on statistical t-maps ('stats') or correlation maps ('corr') as inputs (default = 'corr')
        colormap : cmap
            colormap to use, will be discretized (default = plt.cm.rainbow)
        group_type : str [Only for source]
            defines the type of labels to display for source
            'mode': group labels (mode) 
            'agglom': group labels (agglomerative)    
            'mean': labels obtained from the mean FC or similarity matrix  
        wta_fc_mean : boolean [Only for target]
            set to True to use WTA results coming from mean FC matrix (default = False)
        features : str
            defines if fc ('fc') or similarity matrix ('sim') (i.e., correlations between the fc profiles) is used as features (default = 'fc')
        label_type : str [Only for target]
            defines the type of labels to use to define connectivity patterns (target)
            'indiv': relabeled labels (i.e., specific to each participant)
            'group_mode': group labels (mode) (i.e., same for all participants)
            'group_agglom': group labels (agglomerative) (i.e., same for all participants)       
        order : arr or str
            defines which number is going to be associated with each K value (default = None)
            this is useful as K values do not correspond to segments
            Two options:
                - array containing one value par K, e.g. if K = 7: [1 3 2 7 5 6 4] (if None, we use the sequential order)
                - 'from_file' to indicate that order should be loaded from existing txt file
        slice_y : int
            y position of the slice to display
        show_spinal_levels : boolean
            Defines whether spinal levels are displayed or not (default = True)
        save_figure : boolean
            Set to True to save figure (default = False)

        Outputs
        ------------
        png of the spinal map
            
        '''

        print(f"\033[1mRUN PLOTTING SPINAL MAPS\033[0m")
        print(f"\033[37mK value = {k}\033[0m")
        print(f"\033[37mShowing = {showing}\033[0m")
        print(f"\033[37mFeatures = {features}\033[0m")
        print(f"\033[37mSave figure = {save_figure}\033[0m")
  
        print("The plotting is displayed in neurological orientation (Left > Right)")

        if showing == 'source':
            if group_type is None:
                raise(Exception(f'The parameter group_type is missing!')) 
            elif group_type == 'agglom' or group_type == 'mode':
                path_data = self.config['main_dir'] + self.config['output_dir'] + self.fc_metric + '/' + self.config['output_tag'] + '/source/' + features + '/K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_' + group_type + '_k' + str(k)
            elif group_type == 'mean':
                path_data = self.config['main_dir'] + self.config['output_dir'] + self.fc_metric + '/' + self.config['output_tag'] + '/source/' + features + '/K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_' + indiv_algorithm + '_labels_k' + str(k)
            # Define order
            if order is None:
                order = range(1,k+1)
            elif order == 'from_file':
                if os.path.exists(path_data + '_order.txt'):
                    order = np.loadtxt(path_data + '_order.txt').astype(int)[1:]
                else:
                    raise(Exception(f'Order file could not be found.')) 
            else:
                order = order
            
            if len(order) != k:
                raise(Exception(f'The length of the order information ({len(order)}) is different from K {k}.'))  

        elif showing == 'target':
            if label_type is None and not wta_fc_mean:
                raise(Exception(f'When plotting target, you need to define which labels you want to use!')) 
            if wta_fc_mean:
                path_data = self.config['main_dir'] + self.config['output_dir'] + self.fc_metric + '/' + self.config['output_tag'] + '/target/' + features + '/K' + str(k) + '/wta_fc_mean/K' + str(k) + '_wta'    
            else:
                path_data = self.config['main_dir'] + self.config['output_dir'] + self.fc_metric + '/' + self.config['output_tag'] + '/target/' + features + '/K' + str(k) + '/' + label_type + '_labels' + '/stats/' + indiv_algorithm + '/K' + str(k) + '_wta_' + input_type 
        else:
            raise(Exception(f'The parameter "showing" should be "target" or "source".')) 
    
        if ~hasattr(colormap, 'colors'): # Create a discretized colormap if is not already discrete
            discretized_colormap = ListedColormap(colormap(np.linspace(0, 1, k+1)))
        elif isinstance(colormap.colors, list): # Test if colormap is discrete 
            discretized_colormap = colormap

        # Load data from images 
        img_to_show = nib.load(path_data + '.nii.gz')
        spinal_data= img_to_show.get_fdata().astype(int)

        # Re-assign numbers (only for source, already done for wta maps)
        if showing == 'source': 
            order = np.concatenate(([0], order)) # Add a 0 to the beginning of the re-ordering info (for the background)
            spinal_data = np.take(order, spinal_data)
        
        spinal_data= np.where(spinal_data > 0, spinal_data, np.nan) # To have transparent background  

        # Load template image for background
        template_img = nib.load(self.config['main_dir'] + self.config['template']['spinalcord'])
        template_data = template_img.get_fdata()
        y = template_data.shape[1]//2 if slice_y is None else slice_y # To center in the middle
        plt.imshow(np.rot90(template_data[:,y,:].T,2),cmap='gray');     

        # Load levels if needed
        if show_spinal_levels == True: 
            levels_img = nib.load(self.config['main_dir'] + self.config['spinal_levels'])
            levels_data = levels_img.get_fdata()
            levels_data = np.where(levels_data > 0, levels_data, np.nan) # To have transparent background   
            plt.imshow(np.rot90(levels_data[:,y,:].T,2),cmap='gray');   

        # Plot labels 
        plt.imshow(np.rot90(spinal_data[:,y,:].T,2),cmap=discretized_colormap);   
                                            
        plt.axis('off')

        # If option is set, save results as a png
        if save_figure == True:
            plt.savefig(path_data + '.png')
            if showing == "source":
                np.savetxt(path_data + '_order.txt',order,fmt='%d',delimiter=',')

    def compute_similarity_spinal_levels(self, indiv_algorithm, group_type=None, features='fc', save_figure=True):
        ''' Compute the similarity of the spinal parcellation for K=7 with frostell atlas 
        
        Inputs
        ------------         
        indiv_algorithm : str
            algorithm that was used at the participant level
        group_type : str 
            defines the type of labels to display for source
            'mode': group labels (mode) 
            'agglom': group labels (agglomerative)    
            'mean': labels obtained from the mean FC or similarity matrix  
        features : str
            defines if fc ('fc') or similarity matrix ('sim') (i.e., correlations between the fc profiles) is used as features (default = 'fc')
        save_figure : boolean
            Set to True to save figure (default = True)

        Outputs
        ------------
        
        XX_diag_sim_atlas.txt
            text file containing diagonal of similarity matrix
        XX_sim_atlas.png
            heatmap of similarity matrix
        '''     

        print(f"\033[1mCOMPUTE SIMILARITY WITH ATLAS\033[0m")
        print(f"\033[37mFeatures = {features}\033[0m")

        if group_type is None:
            raise(Exception(f'The parameter group_type is missing!')) 
        elif group_type == 'agglom' or group_type == 'mode':
            path_data = self.config['main_dir'] + self.config['output_dir'] + self.fc_metric + '/' + self.config['output_tag'] + '/source/' + features + '/K7/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_' + group_type + '_k7'
        elif group_type == 'mean':
            path_data = self.config['main_dir'] + self.config['output_dir'] + self.fc_metric + '/' + self.config['output_tag'] + '/source/' + features + '/K7/mean_labels/' + self.config['output_tag'] + '_mean_' + indiv_algorithm + '_labels_k7'
        
        # Load data from images 
        img_to_show = nib.load(path_data + '.nii.gz')
        spinal_data = img_to_show.get_fdata().astype(int)
        # Load atlas
        levels_img = nib.load(self.config['main_dir'] + self.config['spinal_levels'])
        levels_data = levels_img.get_fdata()
        levels_data[~self.mask_source] = 0# To take into account the mask of our analyses

        # We need to convert to 4D data
        k_values = range(1,8)
        spinal_data = (spinal_data[..., np.newaxis] == k_values).astype(int) * spinal_data.reshape(spinal_data.shape + (1,))
        levels_data = (levels_data[..., np.newaxis] == k_values).astype(int) * levels_data.reshape(levels_data.shape + (1,))

        # Compute similarity matrix with atlas
        similarity_matrix,_,orderY = compute_similarity(self.config, spinal_data, levels_data, thresh1=0.1, thresh2=0.1, method='Dice', match_compo=True, verbose=True)   
        print(f'Mean Dice = {np.mean(np.diagonal(similarity_matrix))}')
        np.savetxt(path_data + '_diag_dice_atlas.txt',np.diagonal(similarity_matrix) ,fmt='%2f', delimiter=',')

        # Saving result and figure if applicable
        if save_figure == True:
            sns.heatmap(similarity_matrix,linewidths=.5,square=True,cmap='YlOrBr',vmin=0, vmax=1,xticklabels=orderY+1,yticklabels=np.array(range(1,8)));
            plt.savefig(path_data + '_dice_atlas.pdf', format='pdf')
                
    def plot_loo_validity(self, k_range, to_plot=['dice','ami','ari','vi'], color="black", save_figure=True):
        '''  
        Plot LOO-validity metrics (i.e., Dice between adjacency matrices) to help define the best number of clusters
        
        Inputs
        ------------
        k_range : int, array or range
            number of clusters to plot  
        to_plot: str, list
            indicates internal metrics to plot ('dice', 'ami', 'ari', 'vi') (default = plot all)
        color : str
            color used for plotting (default: 'black')
        save_figures : boolean
            is True, figures are saved as pdf (default = True)
        '''

        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range
        # If only one string is given for internal / group metrics, convert to list
        to_plot = [to_plot] if isinstance(to_plot,str) else to_plot

        # Useful info
        metrics_names = {'dice': 'Dice coefficient',
                         'ami': 'Adjusted Mutual Information ',
                         'ari': 'Adjusted Rand Index',
                         'vi': 'Variation of Information'}
        
        path_validity = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/sim/loo_validity/'
        
        for metric in to_plot:
            loo_k_selection_df = pd.read_pickle(path_validity + self.config['output_tag'] + '_loo_k_selection.pkl')
            # Keep only k of k_range
            loo_k_selection_df  = loo_k_selection_df [loo_k_selection_df ['k'].isin(k_range)] 
            plt.figure(figsize=(5, 3))
            sns.lineplot(data=loo_k_selection_df, x='k', y=metric, errorbar=('ci',95), marker='o', markersize=5, color=color, err_kws={'edgecolor': None, 'alpha': 0.1})  # Plot mean line

            # Set ticks for each k value
            plt.xticks(loo_k_selection_df['k'].unique())

            # Add labels and title
            plt.xlabel('K', fontsize=12, fontweight='bold')
            plt.ylabel(metrics_names[metric], fontsize=12, fontweight='bold')
            plt.tight_layout()

            if save_figure:
                plt.savefig(path_validity + self.config['output_tag'] + '_loo_k_selection.pdf', format='pdf')
                
    def plot_split_half_validity(self, k_range, to_plot=['dice','ami','ari','vi'], color="black", save_figure=True):
        '''  
        Plot LOO-validity metrics (i.e., Dice between adjacency matrices) to help define the best number of clusters
        
        Inputs
        ------------
        k_range : int, array or range
            number of clusters to plot  
        to_plot: str, list
            indicates internal metrics to plot ('dice', 'ami', 'ari', 'vi') (default = plot all)
        color : str
            color used for plotting (default: 'black')
        save_figures : boolean
            is True, figures are saved as pdf (default = True)
        '''

        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range
        # If only one string is given for internal / group metrics, convert to list
        to_plot = [to_plot] if isinstance(to_plot,str) else to_plot

        # Useful info
        metrics_names = {'dice': 'Dice coefficient',
                         'ami': 'Adjusted Mutual Information ',
                         'ari': 'Adjusted Rand Index',
                         'vi': 'Variation of Information'}
        
        path_validity = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/sim/splithalf_validity/'
        
        for metric in to_plot:
            loo_k_selection_df = pd.read_pickle(path_validity + self.config['output_tag'] + '_splithalf_k_selection.pkl')
            # Keep only k of k_range
            loo_k_selection_df  = loo_k_selection_df [loo_k_selection_df ['k'].isin(k_range)] 
            plt.figure(figsize=(5, 3))
            sns.lineplot(data=loo_k_selection_df, x='k', y=metric, errorbar=('ci',95), marker='o', markersize=5, color=color, err_kws={'edgecolor': None, 'alpha': 0.1})  # Plot mean line

            # Set ticks for each k value
            plt.xticks(loo_k_selection_df['k'].unique())

            # Add labels and title
            plt.xlabel('K', fontsize=12, fontweight='bold')
            plt.ylabel(metrics_names[metric], fontsize=12, fontweight='bold')
            plt.tight_layout()

            if save_figure:
                plt.savefig(path_validity + self.config['output_tag'] + '_splithalf_k_selection.pdf', format='pdf')
        
    def plot_validity(self, k_range, internal=[], group=[], take_mean=False, features='fc', indiv_algorithm='kmeans', color="black", save_figures=True):
        '''  
        Plot validity metrics to help define the best number of clusters
        
        Internal metrics are computed during participant-level clustering:
        - SSE (not if 'agglom' is used)
        - Silhouette coefficients
        - Davies-Bouldin index 
        - Calinski-Harabasz index

        Group metrics are computed during group-level clustering:
        - adjusted mutual information (AMI)
        - adjusted rand index (ARI)
        - Cophenetic correlation (how well participant-level distances are preserved in the group-level clustering)

        Inputs
        ------------
        k_range : int, array or range
            number of clusters to plot  
        internal: str, list
            indicates internal metrics to plot ('SSE', 'silhouette', 'davies', 'calinski')
        take_mean : boolean
            if set to True, clustering is done on the mean values (either FC or sim) across all participants
        group : str, list
            indicates group metrics to plot ('ami_agglom', 'ari_agglom', 'ami_mode', 'ari_mode', 'corr')
        features : str
            defines if fc ('fc') or similarity matrix ('sim') (i.e., correlations between the fc profiles) is used as features (default = 'fc')
        indiv_algorithm: str
            defines which algorithm was used ('kmeans' or 'agglom') for participant-level clustering (default = 'kmeans')
        color : str
            color used for plotting (default: 'black')
        save_figures : boolean
            is True, figures are saved as pdf (default = True)
        '''
        
        print(f"\033[1mVALIDITY METRICS\033[0m")
        print(f"\033[37mK value(s) = {k_range}\033[0m")
        print(f"\033[37mFeatures = {features}\033[0m")
        print(f"\033[37mSaving figures = {save_figures}\033[0m\n")

        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range
        
        # If only one string is given for internal / group metrics, convert to list
        internal = [internal] if isinstance(internal,str) else internal
        group = [group] if isinstance(group,str) else group

        # Useful info
        metrics_names = {'SSE': 'SSE',
                         'silhouette': 'Silhouette score',
                         'davies': 'Davies-Bouldin index', 
                         'calinski': 'Calinski-Harabasz index',
                         'ami_mode': 'Adjusted Mutual Information (mode)',
                         'ari_mode': 'Adjusted Rand Index (mode)',
                         'ami_agglom': 'Adjusted Mutual Information (agglom)',
                         'ari_agglom': 'Adjusted Rand Index (agglom)',
                         'corr': 'Cophenetic correlation'}

        # Path to create folder structure
        path_validity = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/' + features + '/validity/'
       
        # Open all dataframes 
        if internal:
            internal_df = pd.read_pickle(path_validity + self.config['output_tag'] + '_' + indiv_algorithm + ''.join(['_internal_validity_mean.pkl' if take_mean else '_internal_validity.pkl']))
        if any(metric in group for metric in ['ami_mode', 'ari_mode', 'ami_agglom', 'ari_agglom']):
            group_df = pd.read_pickle(path_validity + self.config['output_tag'] + '_' + indiv_algorithm + '_group_validity.pkl')
        if any(metric in group for metric in ['corr']):
            corr_df = pd.read_pickle(path_validity + self.config['output_tag'] + '_' + indiv_algorithm + '_cophenetic_correlation.pkl')

        sns.set(style="ticks",  font='sans-serif');

        for metric in (internal + group):
            # Cophenetic correlation as lineplot
            if metric == 'corr':
                data_df = corr_df[corr_df['k'].isin(k_range)]
                plt.figure(figsize=(5, 3))
                sns.lineplot(data=data_df, x='k', y=metric, marker='o', color=color)
                plt.xlabel('K', fontsize=12, fontweight='bold')
                plt.ylabel(metrics_names[metric], fontsize=12, fontweight='bold')
                plt.tight_layout()
                if save_figures:
                    plt.savefig(path_validity + self.config['output_tag'] + '_' + indiv_algorithm + '_' + metric + '.pdf', format='pdf')
            # Other metrics are plotted as boxplots
            else:
                data_df = internal_df[internal_df['k'].isin(k_range)] if metric in internal else group_df[group_df['k'].isin(k_range)]
                plt.figure(figsize=(5, 3))
                if take_mean: # If take mean, we use a line instead of a boxplot
                    sns.lineplot(y=metric, x="k", data=data_df, markersize=5,
                                    linewidth=1, marker='o', color=color);
                else:
                    sns.catplot(y=metric, x="k", data=data_df, kind="box", legend=True, legend_out=True,
                                    linewidth=2,medianprops=dict(color="white"),color=color, 
                                    boxprops=dict(alpha=.6,edgecolor=None),whiskerprops=dict(color=color), capprops=dict(color=color), fliersize=0, aspect=0.5);
                    sns.stripplot(y=metric, x="k", data=data_df, size=5, color=color, alpha=.8, linewidth=0,
                                edgecolor='white', dodge=True)
                     # Add labels and title
                plt.xlabel('K')
                plt.ylabel(metrics_names[metric])
                # Sort and set x-axis tick labels to integers
                plt.xticks(sorted(data_df['k'].unique()))
                plt.tight_layout()
                if save_figures:
                    plt.savefig(path_validity + self.config['output_tag'] + '_' + indiv_algorithm + '_' + metric + ''.join(['_mean.pdf' if take_mean else '.pdf']), format='pdf')
                        
    def _corr2_coeff(self, arr_source, arr_target):
        # Rowwise mean of input arrays & subtract from input arrays themselves
        A_mA = arr_source - arr_source.mean(1)[:, None]
        B_mB = arr_target - arr_target.mean(1)[:, None]

        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)

        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    
    def _compute_mi(self, vox_source, data_source_masked, data_target_masked):
        return mutual_info_regression(data_target_masked.T, data_source_masked[vox_source,:].T, n_neighbors=7)

    def _relabel(self, reference: np.ndarray, x: np.ndarray):
        """Relabel cluster labels to best match a reference
        FROM CBP TOOLBOX"""

        permutations = itertools.permutations(np.unique(x))
        accuracy = 0.
        relabeled = None
    
        for permutation in permutations:
            d = dict(zip(np.unique(x), permutation))
            y = np.zeros(x.shape).astype(int)

            for k, v in d.items():
                y[x == k] = v

            _accuracy = np.sum(y == reference) / len(reference)

            if _accuracy > accuracy:
                accuracy = _accuracy
                relabeled = y.copy()

        return relabeled, accuracy

    def _plot_sim(self,sim_r,save_path,sub):
        """Plot similarity matrix (i.e., correlations of FC profiles)"""
        
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the reordered matrix
        img = ax.imshow(sim_r, cmap='bwr', vmin=-0.5, vmax=0.5)

        # Customize x and y axis ticks
        unique_levels, counts = np.unique(self.levels_sorted, return_counts=True)
        x_label_positions = np.cumsum(counts) - counts / 2
        y_label_positions = x_label_positions

        # Use a dictionary for tick labels
        label_mapping = {1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7'}
        tick_labels = [label_mapping[level] for level in unique_levels]

        # Add vertical/horizontal lines
        line_positions = [np.where(self.levels_sorted == level)[0][-1] for level in unique_levels]
        for x_pos, y_pos in zip(line_positions, line_positions):
            ax.axvline(x_pos, color='black', linestyle=':', linewidth=1)
            ax.axhline(y_pos, color='black', linestyle=':', linewidth=1)

        # Add ticks at the center of each region
        ax.set_xticks(x_label_positions)
        ax.set_yticks(y_label_positions)

        # Set tick labels
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        # Display colorbar
        plt.colorbar(img)

        plt.savefig(save_path + self.config['output_tag'] + '_' + sub + '_sim_reordered.pdf', format='pdf')

    def _plot_pca(self,sim_r,save_path,sub):
        # Perform PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(sim_r)

        # Create a 2D PC plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=self.levels_sorted, cmap='rainbow', marker='.')

        # Set axis labels
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')

        # Set plot title
        ax.set_title('2D PC Plot')

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Levels')

        plt.savefig(save_path + self.config['output_tag'] + '_' +  sub + '_pca.pdf', format='pdf')

    def _cluster_indices_to_adjacency(self, cluster_indices):
        
        # Create a boolean mask where True indicates elements that belong to the same cluster
        adjacency_matrix = (cluster_indices[:, np.newaxis] == cluster_indices[np.newaxis, :]).astype(int)
        
        return adjacency_matrix
    
    def _relabel_labels(self, reference_labels, target_labels):
        # Compute contingency matrix to assess overlap between labels
        cm = contingency_matrix(reference_labels, target_labels)
        # Normalize it to take into account different sizes of clusters
        label_counts = np.bincount(reference_labels)
        # Divide each row of the contingency matrix by the corresponding count
        normalized_cm = cm / label_counts[:, None]
        
        # For each individual label (target), find to which group label (ref) it corresponds best
        cm_argmax = normalized_cm.argmax(axis=0)

        relabel_mapping = {}
        # Loop through the ref labels that have been matched to the individual ones 
        for i in np.unique(cm_argmax):
            # If one group label is the "best" one for multiple indiv labels
            if len(cm_argmax[cm_argmax == i]) > 1: 
                # We check which of the labels has the best overlap
                best = normalized_cm[i,:].argmax()
                relabel_mapping[best] = i
                # For the label(s) that are not mapped, we use -1
                for other_lbl in np.where((cm_argmax == i) & (np.arange(len(cm_argmax)) != best))[0]:
                    relabel_mapping[other_lbl] = -1
            else:
                relabel_mapping[np.argmax(cm_argmax==i)] = i
        
        # Assign relabeled labels using the relabeling mapping
        relabeled_labels = np.zeros_like(target_labels)
        for j, label in enumerate(target_labels):
            if label in relabel_mapping:
                relabeled_labels[j] = relabel_mapping[label]
        
        return relabeled_labels