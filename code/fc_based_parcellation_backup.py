import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from nilearn.maskers import NiftiMasker
import nibabel as nib
from joblib import Parallel, delayed
import time
import os

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
    
    def __init__(self, config, struct_source='spinacord', struct_target='brain', params_kmeans={'init':'k-means++', 'n_init':100, 'max_iter':300}, params_agglom={'linkage':'ward', 'affinity':'euclidean'}):
        self.config = config # Load config info
        self.struct_source = struct_source
        self.struct_target = struct_target
        
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

    def compute_voxelwise_correlation(self, mask_source_path, mask_target_path, load_from_file, save_results=True, Fisher=True, n_jobs=10):
        '''
        To compute Pearson correlation between each voxel of mask_source to all voxels of mask_target
        mask_source/target_path : str
            Paths of masks defining the regions to consider
        load_from_file : boolean
            If set to True, correlations are loaded from file directly
        save_results : boolean
            If set to True, correlations are saved to .npy file (Default = True)
        Fisher : boolean
            To Fisher-transform the correlation (default = True).
        njobs: int
            number of jobs for parallelization (Default = 10)
        '''
        
        print(f"COMPUTE VOXELWISE CORRELATION")
        
        # Read mask data
        self.mask_source_path = mask_source_path
        self.mask_target_path = mask_target_path
        self.mask_source = nib.load(self.mask_source_path).get_data().astype(bool)
        self.mask_target = nib.load(self.mask_target_path).get_data().astype(bool)

        # Compute correlations
        # We can load the correlations from file if it exists
        if load_from_file and os.path.isfile(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_correlations.npy'):
            print(f"... Load correlations from file")
            self.correlations = np.load(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_correlations.npy')
        else: # Otherwise we compute the correlations    
            # start = time.time()
            print(f"... Computing correlations for all possibilities")
            # Create empty array
            self.correlations = np.zeros((len(self.config['list_subjects']),np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_target)))
            for sub_id,sub in enumerate(self.config['list_subjects']):
                print(f"...... Subject {sub}")
                data_source_masked = self.data_source[sub][self.mask_source]
                data_target_masked = self.data_target[sub][self.mask_target] 
                self.correlations[sub_id,:,:] = self._corr2_coeff(data_source_masked,data_target_masked)
            if Fisher == True:
                self.correlations = np.arctanh(self.correlations)
            if save_results == True:            
                np.save(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_correlations.npy',self.correlations)
            #print("... Operation performed in %.2f s!" % (time.time() - start))
            

        print("... Computing mean correlation over subjects")
        #self.correlations_mean = np.mean(np.array([self.correlations[sub] for sub in self.correlations]),axis=0)
        self.correlations_mean = np.mean(self.correlations, axis=0)
        print("DONE!")

    def run_clustering(self, k, algorithm):
        '''  
        Run k-means clustering for a specific number of clusters
        (See define_n_clusters to help pick this)
        
        Input
        ------------
        k: int
            number of clusters  
        algorithm: str
            defines which algorithm to use ('kmeans' or 'agglom')

        Output
        ------------
        kmeans: KMeans
            results of the k_means 
        '''

        self.k = k 
        self.algorithm = algorithm # So that the algorithm is defined based on last run of clustering

        if algorithm == 'kmeans':
            print(f"RUN K-MEANS CLUSTERING FOR K = {self.k}")
            
            # Dict containing k means parameters
            kmeans_kwargs = {'n_clusters': self.k, 'init': self.init, 'max_iter': self.max_iter, 'n_init': self.n_init}

            kmeans_clusters = KMeans(**kmeans_kwargs)
            kmeans_clusters.fit(self.correlations_mean)

            self.clusters = kmeans_clusters
        elif algorithm == 'agglom':
            print(f"RUN AGGLOMERATIVE CLUSTERING FOR K = {self.k}")
            # Dict containing parameters
            agglom_kwargs = {'n_clusters': self.k, 'linkage': self.linkage, 'affinity': self.affinity}

            agglom_clusters = AgglomerativeClustering(**agglom_kwargs)
            agglom_clusters.fit(self.correlations_mean)

            self.clusters = agglom_clusters

        else:
            raise(Exception(f'Algorithm {algorithm} is not a vadid option.'))

        print("DONE")

    def define_n_clusters(self, k_range, algorithm='kmeans', save_results=True):
        '''  
        Probe different number of clusters to define best option
        Two methods are used:
        - SSE for different K
        - Silhouette coefficients
        
        Inputs
        ------------
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
                clusters.fit(self.correlations_mean)
                sse.append(clusters.inertia_)
            elif algorithm == 'agglom':
                clusters = AgglomerativeClustering(n_clusters=k,**kwargs)
                clusters.fit(self.correlations_mean)
                sse = np.zeros(len(k_range)) # SSE not relevant here 
            score = silhouette_score(self.correlations_mean, clusters.labels_)
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
            # Save arrays
            np.save(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_' + algorithm + '_define_K_SSE.npy',sse)
            np.save(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_' + algorithm + '_define_K_silhouette.npy',silhouette_coefficients)
            # Save figure
            plt.savefig(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_' + algorithm + '_define_K.png')

        print("DONE")

    def prepare_seed_map(self):
        '''  
        To obtain images of the labels in the seed region

        Outputs
        ------------
        nifti image
            image in which each voxel is labeled based on the results
            from the clustering (i.e., based on the connectivity profile)
            (e.g., "labels_K4.nii.gz" for 4 clusters)
            
        '''
            
        print("PREPARE SEED MAP")

        seed = NiftiMasker(self.mask_source_path).fit()
        labels_img = seed.inverse_transform(self.clusters.labels_+1) # +1 because labels start from 0 
        labels_img.to_filename(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_' + self.algorithm + '_labels_K' + str(self.k) + '.nii.gz')

        print("DONE")

    def prepare_target_maps(self):
        '''  
        To obtain images of the connectivity profiles assigned to each label
        (i.e., mean over the connectivity profiles of the voxel of this K)

        Outputs
        ------------
        K nifti images
            one image for each mean connectivity profile (i.e., one per K)
            (e.g., "brain_pattern_K4_1.nii.gz" for the first cluster out of 4 total clusters)
            
        '''
            
        print("PREPARE BRAIN MAPS")
        
        print("...Compute mean connectivity profiles")

        brain_maps = np.zeros((len(np.unique(self.clusters.labels_)), self.correlations_mean.shape[1]))
        for label in np.unique(self.clusters.labels_):
            brain_maps[label,:] = np.mean(self.correlations_mean[np.where(self.clusters.labels_==label),:],axis=1)

        print("...Save as nifti files")
        brain_mask = NiftiMasker(self.mask_target_path).fit()
        
        for label in np.unique(self.clusters.labels_):
            brain_map_img = brain_mask.inverse_transform(brain_maps[label,:])
            brain_map_img.to_filename(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_' + self.algorithm + '_brain_pattern_K' + str(self.k) + '_' + str(label+1) + '.nii.gz')

        print("DONE")

    def _corr2_coeff(self, arr_source, arr_target):
        # Rowwise mean of input arrays & subtract from input arrays themeselves
        A_mA = arr_source - arr_source.mean(1)[:, None]
        B_mB = arr_target - arr_target.mean(1)[:, None]

        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)

        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


    def _compute_k_scores(self, k, algorithm, kwargs):
        if algorithm == 'kmeans':
            clusters = KMeans(n_clusters=k,**kwargs)
        elif algorithm == 'agglom':
            clusters = AgglomerativeClustering(n_clusters=k,**kwargs)
        clusters.fit(self.correlations_mean)
        sse = clusters.inertia_
        silhouette = silhouette_score(self.correlations_mean, clusters.labels_)
        return sse, silhouette