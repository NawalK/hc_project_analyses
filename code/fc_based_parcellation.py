import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_mutual_info_score, adjusted_rand_score
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy import stats
import numpy as np
import itertools
from nilearn.maskers import NiftiMasker
import nibabel as nib
import seaborn as sns
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import json
import pandas as pd
import time

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
        - kernel: Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels. Ignored for metric='nearest_neighbors'. (default = 'nearest_neighbors')
        - assign_labels: trategy for assigning labels in the embedding space (default = 'kmeans')
        - eigen_solver: eigenvalue decomposition strategy to use (default = 'arpack')
        - eigen_tol: stopping criterion for eigendecomposition of the Laplacian matrix (default = 1.0e-5)
    params_agglom : dict
        parameters for agglomerative clustering
        - linkage: distance to use between sets of observations (default = 'ward')
        - metric: metric used to compute the linkage (default = 'euclidean')
    '''
    
    def __init__(self, config, struct_source='spinalcord', struct_target='brain', fc_metric='corr', params_kmeans={'init':'k-means++', 'n_init':256, 'max_iter':10000}, params_spectral={'n_init':256, 'kernel': 'nearest_neighbors', 'assign_labels': 'kmeans', 'eigen_solver': 'arpack', 'eigen_tol': 1.0e-5}, params_agglom={'linkage':'ward', 'metric':'euclidean'}):
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
        self.kernel = params_spectral.get('kernel')
        self.assign_labels = params_spectral.get('assign_labels')
        self.eigen_solver = params_spectral.get('eigen_solver')
        self.eigen_tol = params_spectral.get('eigen_tol')

        # Read mask data
        self.mask_source_path = self.config['main_dir']+self.config['masks'][struct_source]
        self.mask_target_path = self.config['main_dir']+self.config['masks'][struct_target]
        self.mask_source = nib.load(self.mask_source_path).get_data().astype(bool)
        self.mask_target = nib.load(self.mask_target_path).get_data().astype(bool)

        # Create folder structure and save config file as json for reference
        path_to_create = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/'
        os.makedirs(os.path.dirname(path_to_create), exist_ok=True)
        for folder in ['fcs', 'source', 'target']:
            os.makedirs(os.path.join(path_to_create, folder), exist_ok=True)
        path_config = path_to_create + 'config_' + self.config['output_tag'] + '.json'
        with open(path_config, 'w') as f:
            json.dump(self.config,f)
            
    def compute_voxelwise_fc(self, sub, standardize=True, overwrite=False, njobs=10):
        '''
        To compute functional connectivity between each voxel of mask_source to all voxels of mask_target
        Can be done using Pearson correlation or Mutual Information
        
        Inputs
        ------------
        sub : str 
            subject on which to run the correlations
        standardize : bool, optional
            if set to True, timeseries are z-scored (defautl = True)
        overwrite : boolean
            if set to True, labels are overwritten (default = False)
        njobs: int
            number of jobs for parallelization [for MI only] (default = 40)

        Output
        ------------
        fc : array
            one array per subject (n_source_voxels x n_target_voxels), saved as a .npy file
        '''
        
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
                
            np.save(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '.npy',fc)
        
        print("\n\033[1mDONE\033[0m")

    def run_clustering(self, sub, k_range, algorithm, poscorr=False, overwrite=False):
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
        k_range : int, array or range
            number of clusters  
        algorithm : str
            defines which algorithm to use ('kmeans', 'spectral', or 'agglom')
        poscorr : boolean
            defines if we take only the positive correlation for the clustering
        overwrite : boolean
            if set to True, labels are overwritten (default = False)
        
        Output
        ------------
        dict_clustering_indiv :  dict
            labels corresponding to the clustering for each k (e.g., dict_clustering_indiv['5'] contains labels of a particular subject for k=5)
            Note: there is one dict per subject, saved as a .pkl file for easy access
        '''

        print(f"\033[1mCLUSTERING AT THE INDIVIDUAL LEVEL\033[0m")
        print(f"\033[37mAlgorithm = {algorithm}\033[0m")
        print(f"\033[37mK value(s) = {k_range}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        if poscorr:
            if self.fc_metric == 'corr':
                print(f"\033[37mNote: keeping only positive correlations\033[0m")
            else:
                raise(Exception(f'The option to keep only positive correlations cannot be used with this metric.'))
        
        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range

        # Path to create folder structure
        path_source = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/'
        os.makedirs(os.path.join(path_source, 'validity'), exist_ok=True)
        path_internal_validity = path_source + 'validity/' + self.config['output_tag'] + '_' + algorithm + '_internal_validity_poscorr.pkl' if poscorr else path_source + 'validity/' + self.config['output_tag'] + '_' + algorithm + '_internal_validity.pkl'
        # Load file with metrics to define K (SSE, silhouette, ...)
        if os.path.isfile(path_internal_validity): # Take it if it exists
            internal_validity_df = pd.read_pickle(path_internal_validity)
            if overwrite: # Overwrite subject if already done and option is set
                internal_validity_df = internal_validity_df[internal_validity_df['sub'] != sub]
        else: # Create an empty dataframe with the needed columns
            columns = ["sub", "SSE", "silhouette", "davies", "calinski", "k"]
            internal_validity_df = pd.DataFrame(columns=columns)

        # Check if file already exists
        for k in k_range:
            print(f"K = {k}")
            
            # Create folder structure if needed
            for folder in ['indiv_labels','indiv_labels_relabeled','group_labels']:
                os.makedirs(os.path.join(path_source, 'K'+str(k), folder), exist_ok=True)

            # Check if file already exists
            path_indiv_labels = path_source + 'K' + str(k) + '/indiv_labels/' + self.config['output_tag'] + '_' + sub + '_' + algorithm + '_labels_k' + str(k) + '_poscorr.npy' if poscorr else path_source + 'K' + str(k) + '/indiv_labels/' + self.config['output_tag'] + '_' + sub + '_' + algorithm + '_labels_k' + str(k) + '.npy'
            if not overwrite and os.path.isfile(path_indiv_labels):
                print(f"... Labels already computed")
            
            # Otherwise, we compute them
            else:
                print(f"... Loading FC from file")
                path_fc = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '.npy'
                if os.path.isfile(path_fc):
                    fc = np.load(path_fc)
                    if poscorr:
                        if self.fc_metric == 'corr':
                            fc[fc<0] = 0; # Keep only positive correlations 
                        else:
                            raise(Exception(f'The option to keep only positive correlations cannot be used with this metric.'))
                else:
                    raise(Exception(f'FC cannot be found.'))
                
                if algorithm == 'kmeans':
                    print(f"... Running k-means clustering")
                    # Dict containing k means parameters
                    kmeans_kwargs = {'n_clusters': k, 'init': self.init, 'max_iter': self.max_iter, 'n_init': self.n_init_kmeans}

                    # Compute clustering
                    kmeans_clusters = KMeans(**kmeans_kwargs)
                    kmeans_clusters.fit(fc)
                    labels = kmeans_clusters.labels_

                    # Compute validity metrics and add them to dataframe
                    internal_validity_df.loc[len(internal_validity_df)] = [sub, kmeans_clusters.inertia_, silhouette_score(fc, labels), davies_bouldin_score(fc, labels), calinski_harabasz_score(fc, labels), k]
                
                elif algorithm == 'spectral':
                    print(f"... Running spectral clustering")
                    spectral_kwargs = {'n_clusters': k, 'n_init': self.n_init_spectral, 'metric': self.kernel,
                            'assign_labels': self.assign_labels, 'eigen_solver': self.eigen_solver, 'eigen_tol': self.eigen_tol}
                    
                    spectral_clusters = SpectralClustering(**spectral_kwargs)
                    spectral_clusters.fit(fc)
                    labels = spectral_clusters.labels_
                    
                    internal_validity_df.loc[len(internal_validity_df)] = [sub, 0, silhouette_score(fc, labels), davies_bouldin_score(fc, labels), calinski_harabasz_score(fc, labels), k]

                elif algorithm == 'agglom':
                    print(f"... Running agglomerative clustering")
                    # Dict containing parameters
                    agglom_kwargs = {'n_clusters': k, 'linkage': self.linkage, 'metric': self.metric}

                    agglom_clusters = AgglomerativeClustering(**agglom_kwargs)
                    agglom_clusters.fit(fc)
                    labels = agglom_clusters.labels_

                    # Compute validity metrics and add them to dataframe
                    # Note that SSE is not relevant for this type of clustering
                    internal_validity_df.loc[len(internal_validity_df)] = [sub, 0,  silhouette_score(fc, labels), davies_bouldin_score(fc, labels), calinski_harabasz_score(fc, labels), k]
                    
                else:
                    raise(Exception(f'Algorithm {algorithm} is not a valid option.'))
            
                np.save(path_indiv_labels, labels.astype(int))

        internal_validity_df.to_pickle(path_internal_validity ) 

        print("\n")
       
    def group_clustering(self, k_range, indiv_algorithm, poscorr=False, linkage="complete", overwrite=False):
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
            define type of linkage to use for hierarchical clustering (default = "complete")
        poscorr : boolean
            defines if we take only the positive correlation for the clustering
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
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
       
        if overwrite == False:
            print("\033[38;5;208mWARNING: THESE RESULTS CHANGE IF GROUP CHANGES, MAKE SURE YOU ARE USING THE SAME SUBJECTS\033[0m\n")
        
        if poscorr:
            if self.fc_metric == 'corr':
                print(f"\033[37mNote: keeping only positive correlations\033[0m")
            else:
                raise(Exception(f'The option to keep only positive correlations cannot be used with this metric.'))
        
      
        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range

        path_source = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/'
        path_group_validity = path_source + 'validity/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_validity_poscorr.pkl' if poscorr else path_source + 'validity/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_validity.pkl'
        path_cophenetic_correlation = path_source + 'validity/' + self.config['output_tag'] + '_' + indiv_algorithm + '_cophenetic_correlation_poscorr.pkl' if poscorr else path_source + 'validity/' + self.config['output_tag'] + '_' + indiv_algorithm + '_cophenetic_correlation.pkl'
        
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
            group_labels_mode_path = path_source + 'K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_mode_k' + str(k) + '_poscorr' if poscorr else path_source + 'K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_mode_k' + str(k) 
            group_labels_agglom_path = path_source + 'K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_agglom_k' + str(k) + '_poscorr' if poscorr else path_source + 'K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_agglom_k' + str(k)
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
                    indiv_labels_path =  path_source + '/K' + str(k) + '/indiv_labels/' + self.config['output_tag'] + '_' + sub + '_' + indiv_algorithm + '_labels_k' + str(k) + '_poscorr.npy' if poscorr else path_source + '/K' + str(k) + '/indiv_labels/' + self.config['output_tag'] + '_' + sub + '_' + indiv_algorithm + '_labels_k' + str(k) + '.npy'    
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
                path_relabeled = path_source + 'K' + str(k) + '/indiv_labels_relabeled/' + self.config['output_tag'] + '_' + indiv_algorithm + '_labels_relabeled_k' + str(k) + '_poscorr.npy' if poscorr else path_source + 'K' + str(k) + '/indiv_labels_relabeled/' + self.config['output_tag'] + '_' + indiv_algorithm + '_labels_relabeled_k' + str(k) + '.npy'
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

    def prepare_target_maps(self, label_type, k_range, indiv_algorithm, overwrite=False):
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
        k_range : int, array or range
            number of clusters  
        indiv_algorithm : str
            algorithm that was used at the participant level
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
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")

        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range
        
        # Path to create folder structure
        path_target = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/target/'
       
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
                        labels = np.load(self.config['main_dir'] + self.config['output_dir'] +  '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/source/K' + str(k) + '/indiv_labels_relabeled/' + self.config['output_tag'] + '_' + indiv_algorithm + '_labels_relabeled_k' + str(k) + '.npy')
                        labels =  np.squeeze(labels[sub_id,:].T)
                    elif label_type == 'group_mode':
                        labels = np.load(self.config['main_dir'] + self.config['output_dir'] +  '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/source/K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_mode_k' + str(k) + '.npy')
                    elif label_type == 'group_agglom':
                        labels = np.load(self.config['main_dir'] + self.config['output_dir'] +  '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/source/K' + str(k) + '/group_labels/' + self.config['output_tag'] + '_' + indiv_algorithm + '_group_labels_agglom_k' + str(k) + '.npy')
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
                        target_map_img.to_filename(path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/maps_indiv' + '/' +  self.config['output_tag'] + '_' + sub + '_' + indiv_algorithm + '_' + label_type + '_labels_targetmap_K' + str(k) + '_' + str(label+1) + '.nii.gz')

                # Save mean maps as nifti files
                for label in np.unique(labels):      
                    target_map_mean_img = target_mask.inverse_transform(np.mean(target_maps[:,label,:],axis=0))
                    target_map_mean_img.to_filename(path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/maps_mean' + '/' +  self.config['output_tag'] + '_mean_' + indiv_algorithm + '_' + label_type + '_labels_targetmap_K' + str(k) + '_' + str(label+1) + '.nii.gz')
                    target_map_Z_img = target_mask.inverse_transform(np.mean(target_maps[:,label,:],axis=0)/np.std(target_maps[:,label,:],axis=0))
                    target_map_Z_img.to_filename(path_target + '/K' + str(k) + '/' + label_type + '_labels' + '/maps_mean' + '/' +  self.config['output_tag'] + '_Z_' + indiv_algorithm + '_' + label_type + '_labels_targetmap_K' + str(k) + '_' + str(label+1) + '.nii.gz')

                # Save array
                np.save(target_npy_path, target_maps)
            
        print("\033[1mDONE\033[0m\n")

    def plot_validity(self, k_range, internal=[], group=[], indiv_algorithm='kmeans', save_figures=True):
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
        group : str, liit
            indicates group metrics to plot ('ami_agglom', 'ari_agglom', 'ami_mode', 'ari_mode', 'corr')
        indiv_algorithm: str
            defines which algorithm was used ('kmeans' or 'agglom') for participant-level clustering (default = 'kmeans')
        save_figures : boolean
            is True, figures are saved as pdf (default = True)
        '''
        
        print(f"\033[1mVALIDITY METRICS\033[0m")
        print(f"\033[37mK value(s) = {k_range}\033[0m")
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
        color="#43c7cc"

        # Path to create folder structure
        path_validity = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/source/validity/'
       
        # Open all dataframes 
        if internal:
            internal_df = pd.read_pickle(path_validity + self.config['output_tag'] + '_' + indiv_algorithm + '_internal_validity.pkl')
        if any(metric in group for metric in ['ami_mode', 'ari_mode', 'ami_agglom', 'ari_agglom']):
            group_df = pd.read_pickle(path_validity + self.config['output_tag'] + '_' + indiv_algorithm + '_group_validity.pkl')
        if any(metric in group for metric in ['corr']):
            corr_df = pd.read_pickle(path_validity + self.config['output_tag'] + '_' + indiv_algorithm + '_cophenetic_correlation.pkl')

        sns.set(style="ticks",  font='sans-serif');

        for metric in (internal + group):
            # Cophenetic correlation as lineplot
            if metric == 'corr':
                data_df = corr_df[corr_df['k'].isin(k_range)]
                plt.figure()
                sns.lineplot(data=data_df, x='k', y=metric, marker='o', color=color)
                plt.xlabel('K', fontsize=12, fontweight='bold')
                plt.ylabel(metrics_names[metric], fontsize=12, fontweight='bold')
            # Other metrics are plotted as boxplots
            else:
                data_df = internal_df[internal_df['k'].isin(k_range)] if metric in internal else group_df[group_df['k'].isin(k_range)]
                g = sns.catplot(y=metric, x="k", data=data_df, kind="box", legend=True, legend_out=True,
                                linewidth=2,medianprops=dict(color="white"),color=color, 
                                boxprops=dict(alpha=.6,edgecolor=None),whiskerprops=dict(color=color), capprops=dict(color=color), fliersize=0, aspect=0.5);
                g.fig.set_size_inches(5, 4)
                sns.stripplot(y=metric, x="k", data=data_df, size=5, color=color, alpha=.8, linewidth=0,
                            edgecolor='white', dodge=True)
                g.set_axis_labels("K", metrics_names[metric], fontsize=12, fontweight='bold')

                if save_figures:
                    g.savefig(path_validity + self.config['output_tag'] + '_' + indiv_algorithm + '_' + metric + '.pdf', format='pdf')
                        
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