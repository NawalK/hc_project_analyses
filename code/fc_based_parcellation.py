import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from scipy import stats
import numpy as np
from nilearn.maskers import NiftiMasker
import nibabel as nib
import itertools
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import json

class FC_Parcellation:
    '''
    The FC_Parcellation class is used to perform the parcellation of a specific roi
    based on the FC profiles of each of its voxels
    
    Attributes
    ----------
    config : dict
    struct_source/data_target : str
        Source (i.e., to parcellate) and target (i.e., the one with wich each voxel of the source is correlated) structures
        Default is struct_source = 'spinalcord', struct_target = 'brain'
    fc_metric: str
        Defines metrics to used to compute functional connectivity (e.g., 'corr' or 'mi') (Default = 'corr')
    clusters : dict
        Will contain the labeling results for each subject
    data_source/target : dict of array
        Contains 4D data of all subjects for the structures of interest
    params_kmeans : dict
        Parameters for k-means clustering
        - init: method for initialization (Default = 'k-means++')
        - n_init: number of times the algorithm is run with different centroid seeds (Default = 100)
        - max_iter: maximum number of iterations of the k-means algorithm for a single run (Default = 300)
    params_agglom : dict
        Parameters for agglomerative clustering
        - linkage: distance to use between sets of observations (Default = 'ward')
        - affinity: metric used to compute the linkage (Default = 'euclidean')
    '''
    
    def __init__(self, config, struct_source='spinalcord', struct_target='brain', fc_metric='corr', params_kmeans={'init':'k-means++', 'n_init':100, 'max_iter':300}, params_agglom={'linkage':'ward', 'affinity':'euclidean'}):
        self.config = config # Load config info
        self.struct_source = struct_source
        self.struct_target = struct_target
        self.fc_metric=fc_metric
        self.clusters = {}
        # Load data
        self.data_source = {}
        self.data_target = {}
        for sub in self.config['list_subjects']:
            self.data_source[sub] = nib.load(config['main_dir'] + config['smooth_dir'] + 'sub-'+ sub + '/' + self.struct_source + '/sub-' + sub + config['coreg_tag'][struct_source] + '.nii.gz').get_fdata() # Read the data as a matrix
            self.data_target[sub] = nib.load(config['main_dir'] + config['smooth_dir'] + 'sub-'+ sub + '/' + self.struct_target + '/sub-' + sub + config['coreg_tag'][struct_target] + '.nii.gz').get_fdata() 

        self.init = params_kmeans.get('init')
        self.n_init = params_kmeans.get('n_init')
        self.max_iter = params_kmeans.get('max_iter')
        self.linkage = params_agglom.get('linkage')
        self.affinity = params_agglom.get('affinity')

        # Create folder structure and save config file as json for reference
        path_to_create = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/'
        os.makedirs(os.path.dirname(path_to_create), exist_ok=True)
        for folder in ['labels', 'target']:
            os.makedirs(os.path.join(path_to_create, folder, 'arrays'), exist_ok=True)
            os.makedirs(os.path.join(path_to_create, folder, 'maps'), exist_ok=True)
        path_config = path_to_create + 'config_' + self.config['output_tag'] + '.json'
        with open(path_config, 'w') as f:
            json.dump(self.config,f)
            
    def compute_voxelwise_fc(self, sub, mask_source_path, mask_target_path, load_from_file, save_results=True):
        '''
        To compute functional connectivity between each voxel of mask_source to all voxels of mask_target
        Can be done using Pearson correlation or Mutual Information
        
        Inputs
        ------------
        sub : str 
            subject on which to run the correlations
        mask_source/target_path : str
            paths of masks defining the regions to consider
        load_from_file : boolean
            if set to True, fc loaded from file directly
        save_results : boolean
            if set to True, fc saved to .npy file (Default = True)
        njobs: int
            number of jobs for parallelization (Default = 10)

        Output
        ------------
        dict_fc : dict
            dictionary containing two fields
            id -> subject tag for saving (e.g., name, 'mean', ...)
            connectivity -> matrix to use as input for the clustering (typically an array with dimensions n_sc_voxels x n_br_voxels)
        '''
        
        print(f"COMPUTE VOXELWISE FC")
        
        # Read mask data
        self.mask_source_path = mask_source_path
        self.mask_target_path = mask_target_path
        self.mask_source = nib.load(self.mask_source_path).get_data().astype(bool)
        self.mask_target = nib.load(self.mask_target_path).get_data().astype(bool)

        # Compute FC
        # We can load it from file if it exists
        if load_from_file and os.path.isfile(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '.npy'):
            print(f"... Load FC from file")
            fc = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '.npy')
        else: # Otherwise we compute FC    
            print(f"... Computing FC for all possibilities")
            
            # Create empty array
            fc = np.zeros((np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_target)))
            data_source_masked = self.data_source[sub][self.mask_source]
            data_target_masked = self.data_target[sub][self.mask_target] 
    
            if self.fc_metric == 'corr':
                print("... Metric: correlation")
                fc = self._corr2_coeff(data_source_masked,data_target_masked)
                print(f"... Fisher transforming correlations")
                fc = np.arctanh(fc)
            elif self.fc_metric == 'mi':
                print("... Metric: mutual information")
                #for vox_source in tqdm(range(np.shape(data_source_masked)[0])):
                #    fc[vox_source,:] = mutual_info_regression(data_target_masked.T, data_source_masked[vox_source,:].T, n_neighbors=8) 
                #    fc[vox_source,:] = np.max(fc[vox_source,:])
                pool = Pool(40)
                vox_source_list = list(range(np.shape(data_source_masked)[0]))
                result_list = []
                for result in tqdm(pool.imap_unordered(partial(self._compute_mi, data_source_masked=data_source_masked, data_target_masked=data_target_masked), vox_source_list), total=len(vox_source_list)):
                    result_list.append(result)
                pool.close()
                pool.join()
                for i, result in enumerate(result_list):
                    fc[i, :] = result
                    fc[i, :] /= np.max(fc[i, :])
            
            if save_results == True:            
                np.save(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/' + self.config['output_tag'] + '_' + sub + '_' + self.fc_metric + '.npy',fc)
            
        dict_fc = {}
        dict_fc['id'] = sub
        dict_fc['connectivity'] = fc
        
        return dict_fc

    def run_clustering(self, dict_fc, k, algorithm, load_from_file, save_results=True):
        '''  
        Run clustering for a specific number of clusters
        (See define_n_clusters to help pick this)
        
        Inputs
        ------------
        dict_fc: dict
            dictionary containing two fields
            id -> subject tag for saving (e.g., name, 'mean', ...)
            connectivity -> matrix to use as input for the clustering (typically an array with dimensions n_sc_voxels x n_br_voxels)
        k : int
            number of clusters  
        algorithm : str
            defines which algorithm to use ('kmeans' or 'agglom')
        load_from_file : boolean
            if set to True, labels loaded from file directly
        save_results : boolean
            if set to True, cluster labels are saved to .npy file (Default = True)
        
        Output
        ------------
        labels :  arr
            labels corresponding to the clustering 
        '''

        self.k = k 
        self.algorithm = algorithm # So that the algorithm is defined based on last run of clustering

        if load_from_file and os.path.isfile(self.config['main_dir'] + self.config['output_dir'] +  '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/labels/arrays/' + self.config['output_tag'] + '_' + dict_fc['id'] + '_' + algorithm + '_labels_k' + str(self.k) + '.npy'):
            print(f"... Load labels from file")
            labels = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/labels/arrays/' + self.config['output_tag'] + '_' + dict_fc['id'] + '_' + algorithm + '_labels_k' + str(self.k) + '.npy')
        
        else:
            if algorithm == 'kmeans':
                print(f"RUN K-MEANS CLUSTERING FOR K = {self.k}")
                
                # Dict containing k means parameters
                kmeans_kwargs = {'n_clusters': self.k, 'init': self.init, 'max_iter': self.max_iter, 'n_init': self.n_init}

                kmeans_clusters = KMeans(**kmeans_kwargs)
                kmeans_clusters.fit(dict_fc['connectivity'])

                labels = kmeans_clusters.labels_

            elif algorithm == 'agglom':
                print(f"RUN AGGLOMERATIVE CLUSTERING FOR K = {self.k}")
                # Dict containing parameters
                agglom_kwargs = {'n_clusters': self.k, 'linkage': self.linkage, 'affinity': self.affinity}

                agglom_clusters = AgglomerativeClustering(**agglom_kwargs)
                agglom_clusters.fit(dict_fc['connectivity'])

                labels = agglom_clusters.labels_

            else:
                raise(Exception(f'Algorithm {algorithm} is not a vadid option.'))

            if save_results == True:
                # Save arrays
                np.save(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/labels/arrays/' + self.config['output_tag'] + '_' + dict_fc['id'] + '_' + algorithm + '_labels_k' + str(self.k) + '.npy', labels)
                
        return labels

    def group_clustering(self,indiv_labels,linkage="complete"):
        '''  
        Perform group-level clustering using the individual labels
        BASED ON CBP toolbox

        Input
        ------------
        indiv_labels : array
            labels for each individual and source voxels (dimension: nb_sujects x nb_voxel_source)

        Output
        ------------
        group_labels : array
            labels for the group
        '''
        print(f"RUN GROUP CLUSTERING")

        # Hierarchical clustering on all labels
        x = indiv_labels.T
        y = pdist(x, metric='hamming')
        z = hierarchy.linkage(y, method=linkage, metric='hamming')
        group_labels = hierarchy.cut_tree(z, n_clusters=len(np.unique(x)))
        group_labels = np.squeeze(group_labels)  # (N, 1) to (N,)

        # Use the hierarchical clustering as a reference to relabel individual
        # participant clustering results
        relabeled = np.empty((0, indiv_labels.shape[1]), int)
        accuracy = []

        # iterate over individual participant labels (rows)
        for label in indiv_labels:
            x, acc = self._relabel(reference=group_labels, x=label)
            relabeled = np.vstack([relabeled, x])
            accuracy.append(acc)

        indiv_labels_relabeled = relabeled
        
        mode, _ = stats.mode(indiv_labels_relabeled, axis=0, keepdims=False)

        # Set group labels to mode for mapping
        group_labels = np.squeeze(mode)
        print("DONE")
        return group_labels, indiv_labels_relabeled

    def define_n_clusters(self, dict_fc, k_range, algorithm='kmeans', save_results=True):
        '''  
        Probe different number of clusters to define best option
        Two methods are used:
        - SSE for different K
        - Silhouette coefficients
        
        Inputs
        ------------
        dict_fc : dict
            dictionary containing two fields
            id -> subject tag for saving (e.g., name, 'mean', ...)
            connectivity -> matrix to use as input for the clustering (typically an array with dimensions n_sc_voxels x n_br_voxels)
        k_range : array
            Number of clusters to include in the comparison
        algorithm: str
            defines which algorithm to use ('kmeans' or 'agglom') (Default = 'kmeans')
        save_results : boolean
            Results are saved as npy and png if set to True (Default = True)
         
        '''
        
        print("DEFINE NUMBER OF CLUSTERS")

        print(f"...Loading clustering parameters, method {algorithm}")
  
        # Dict containing clustering parameters
        if algorithm == 'kmeans':
            kwargs = {'init': self.init, 'max_iter': self.max_iter, 'n_init': self.n_init}     
        elif algorithm == 'agglom':
            kwargs = {'linkage': self.linkage, 'affinity': self.affinity}
        else:
            raise(Exception(f'Algorithm {algorithm} is not a vadid option.'))

        sse = []
        silhouette_coefficients = []

        print("...Computing SSE and silhouette coefficients")
        # Compute metrics for each number of clusters
        for k in k_range:
            print(f"......K = {k}")

            if algorithm == 'kmeans':
                clusters = KMeans(n_clusters=k,**kwargs)
                clusters.fit(dict_fc['connectivity'])
                sse.append(clusters.inertia_)
            elif algorithm == 'agglom':
                clusters = AgglomerativeClustering(n_clusters=k,**kwargs)
                clusters.fit(dict_fc['connectivity'])
                sse = np.zeros(len(k_range)) # SSE not relevant here 
            score = silhouette_score(dict_fc['connectivity'], clusters.labels_)
            silhouette_coefficients.append(score)

        # Find knee point of SSE
        kl = KneeLocator(k_range, sse, curve="convex", direction="decreasing")
        sse_knee = kl.elbow
        print(f'Knee of SSE curve is at K = {sse_knee}')    
        # Two subplots
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 3))
        # SSE subplot with knee highlighted
        ax1.plot(k_range, sse)
        ax1.set(xticks=k_range, xlabel = "Number of Clusters", ylabel = "SSE")
        if sse_knee is not None: 
            ax1.axvline(x=sse_knee, color='r')
        # Silhouette coefficient
        ax2.plot(k_range, silhouette_coefficients)
        ax2.set(xticks=k_range, xlabel = "Number of Clusters", ylabel = "Silhouette Coefficient")
        fig.tight_layout()

        if save_results == True:
            path_to_create = self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/define_K/'
            os.makedirs(os.path.dirname(path_to_create), exist_ok=True)
            # Save arrays
            np.save(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/define_K/' + self.config['output_tag'] + '_' + dict_fc['id'] + '_' + algorithm + '_define_K_SSE.npy',sse)
            np.save(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/define_K/' + self.config['output_tag'] + '_' + dict_fc['id'] + '_' + algorithm + '_define_K_silhouette.npy',silhouette_coefficients)
            # Save figure
            plt.savefig(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/define_K/' + self.config['output_tag'] + '_' + dict_fc['id'] + '_' + algorithm + '_define_K.png')

        print("DONE")

    def prepare_seed_map(self, labels):
        '''  
        To obtain images of the labels in the seed region

        Input
        ------------
        labels : array
            labels to put in nifti
        
        Outputs
        ------------
        nifti image
            image in which each voxel is labeled based on the results
            from the clustering (i.e., based on the connectivity profile)
            (e.g., "labels_K4.nii.gz" for 4 clusters)
            
        '''
            
        print("PREPARE SEED MAP")

        seed = NiftiMasker(self.mask_source_path).fit()
        labels_img = seed.inverse_transform(labels+1) # +1 because labels start from 0 
        labels_img.to_filename(self.config['main_dir'] + self.config['output_dir']  + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/labels/maps/' + self.config['output_tag'] + '_' + self.algorithm + '_labels_k' + str(self.k) + '.nii.gz')

        print("DONE")

    def prepare_target_maps(self, labels, load_from_file, save_results=True):
        '''  
        To obtain images of the connectivity profiles assigned to each label
        (i.e., mean over the connectivity profiles of the voxel of this K)

        Inputs
        ------------
        labels : array
            Labels (source) to use to define connectivity patterns (target) (nb_subjects x nb_vox_source)
            Normally, relabeled labelled of each subject
        load_from_file : boolean
            if set to True, maps loaded from file directly
        save_results : boolean
            if set to True, maps are saved to .npy file (Default = True)
        
        Outputs
        ------------
        target_maps : array
            array containing the brain maps for each label and subject (nb_subjects x K x n_vox_target)
        K nifti images
            one image for each mean connectivity profile (i.e., one per K)
            (e.g., "brain_pattern_K4_1.nii.gz" for the first cluster out of 4 total clusters)
            
        '''
            
        print("PREPARE TARGET MAPS")
        if load_from_file and os.path.isfile(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/target/arrays/' + self.config['output_tag'] + '_target_maps.npy'):
            print(f"... Load target maps from file")
            target_maps = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/target/arrays/' + self.config['output_tag'] + '_target_maps.npy')
            
        else:
            # Initialize array to save target data
            target_maps = np.zeros((labels.shape[0],len(np.unique(labels)),np.count_nonzero(self.mask_target)))
            for sub in range(0,labels.shape[0]):
                print(f"Subject {self.config['list_subjects'][sub]}")
                print(f"... Load FC")
                fc = np.load(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric + '/' + self.config['output_tag'] + '/' + self.config['output_tag'] + '_' + self.config['list_subjects'][sub] + '_' + self.fc_metric + '.npy')
                print(f"... Compute mean map per label")
                for label in np.unique(labels):
                    target_maps[sub,label,:] = np.mean(fc[np.where(labels[sub,:]==label),:],axis=1)
                    
                print("... Save as nifti files")
                target_mask = NiftiMasker(self.mask_target_path).fit()
                
                for label in np.unique(labels):
                    if os.path.isfile(self.config['main_dir'] + self.config['output_dir']  + '/' + self.fc_metric  + '/'  + self.config['output_tag'] + '/' + self.config['output_tag'] + '/target/maps/' + self.config['output_tag'] + '_sub_' + self.config['list_subjects'][sub] + '_' + self.algorithm + '_brain_pattern_K' + str(self.k) + '_' + str(label+1) + '.nii.gz'):
                        raise(Exception(f'Maps already exist!'))
                    else:
                        target_map_img = target_mask.inverse_transform(target_maps[sub,label,:])
                        target_map_img.to_filename(self.config['main_dir'] + self.config['output_dir']  + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/target/maps/' + self.config['output_tag'] + '_sub_' + self.config['list_subjects'][sub] + '_' + self.algorithm + '_brain_pattern_K' + str(self.k) + '_' + str(label+1) + '.nii.gz')
            
            if save_results == True:
                # Save array
                np.save(self.config['main_dir'] + self.config['output_dir'] + '/' + self.fc_metric  + '/' + self.config['output_tag'] + '/target/arrays/' + self.config['output_tag'] + '_target_maps.npy', target_maps)
         
        
        print("DONE")

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
        return mutual_info_regression(data_target_masked.T, data_source_masked[vox_source,:].T, discrete_features=True, n_neighbors=8)

    def _compute_k_scores(self, k, algorithm, kwargs):
        if algorithm == 'kmeans':
            clusters = KMeans(n_clusters=k,**kwargs)
        elif algorithm == 'agglom':
            clusters = AgglomerativeClustering(n_clusters=k,**kwargs)
        clusters.fit(self.correlations_mean)
        sse = clusters.inertia_
        silhouette = silhouette_score(self.correlations_mean, clusters.labels_)
        return sse, silhouette

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