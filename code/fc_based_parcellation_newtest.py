import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from nilearn.maskers import NiftiMasker
import nibabel as nib
import time

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
        Parameters for the k-means clustering
        - init: method for initialization (Default = 'k-means++')
        - n_init: number of times the algorithm is run with different centroid seeds (Default = 100)
        - max_iter: maximum number of iterations of the k-means algorithm for a single run (Default = 300)
    '''
    
    def __init__(self, config, struct_source='spinacord', struct_target='brain', params_kmeans={'init':'k-means++', 'n_init':100, 'max_iter':300}):
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

        self.correlations = {}

    def compute_voxelwise_correlation(self, mask_source_path, mask_target_path, subsample_target=True, Fisher=True, n_jobs=10):
        '''
        To compute Pearson correlation between each voxel of mask_source to all voxels of mask_target
        mask_source/target_path : str
            Paths of masks defining the regions to consider
        subsample_target : boolean
            If set to True, we take only one out of two elements in the target mask, to make computations faster (Default = True)
        Fisher : bool
            To Fisher-transform the correlation (default = True).
        njobs: int
            number of jobs for parallelization (Default = 10)
        '''
        
        print(f"COMPUTE VOXELWISE CORRELATION")
        
        # Read mask data
        self.mask_source = nib.load(mask_source_path).get_data().astype(bool)
        self.mask_target = nib.load(mask_target_path).get_data().astype(bool)

        # Subsample target masks if needed
        if subsample_target == True:
            print(f"... Subsampling target mask for efficiency")
            subsampling_mask = np.zeros(self.mask_target.shape).astype(bool)
            subsampling_mask[::2,::2,::2] = 1
            self.mask_target *= subsampling_mask

        # Compute correlations
        # start = time.time()
        print(f"... Computing correlations for all possibilities")
        for sub in self.config['list_subjects']:
            print(f"...... Subject {sub}")
            data_source_masked = self.data_source[sub][self.mask_source]
            data_target_masked = self.data_target[sub][self.mask_target] 
            self.correlations[sub] = self._corr2_coeff(data_source_masked,data_target_masked)
        
        #print("... Operation performed in %.2f s!" % (time.time() - start))
        print("... Computing mean correlation over subjects")
        self.correlations_mean = np.mean(np.array([self.correlations[sub] for sub in self.correlations]),axis=0)
        print("DONE!")

    def run_clustering(self,k):
        '''  
        Run k-means clustering for a specific number of clusters
        (See define_n_clusters to help pick this)
        
        Input
        ------------
        k: int
            number of clusters  

        Output
        ------------
        kmeans: KMeans
            results of the k_means 
        '''

        self.k = k 

        print(f"RUN K-MEANS CLUSTERING FOR K = {self.k}")
        
        # Dict containing k means parameters
        kmeans_kwargs = { 'init': self.init, 'max_iter': self.max_iter, 'n_init': self.n_init}

        kmeans = KMeans(n_clusters=self.k,**kmeans_kwargs)
        kmeans.fit(self.correlations_mean)

        self.kmeans = kmeans

        print("DONE")

    def define_n_clusters(self, k_range):
        '''  
        Probe different number of clusters to define best option
        Two methods are used:
        - SSE for different K
        - Silhouette coefficients
        
        Inputs
        ------------
        k_range : array
            Number of clusters to include in the comparison
     
        '''
        
        print("DEFINE NUMBER OF CLUSTERS")

        print("...Loading k-means parameters")
  
        # Dict containing k means parameters
        kmeans_kwargs = {'init': self.init, 'max_iter': self.max_iter, 'n_init': self.n_init}

        sse = []
        silhouette_coefficients = []

        print("...Computing SSE and silhouette coefficients")
        # Compute metrics for each number of clusters
        for k in k_range:
            print(f"......K = {k}")
            kmeans = KMeans(n_clusters=k,**kmeans_kwargs)
            kmeans.fit(self.correlations_mean)
            sse.append(kmeans.inertia_)
            score = silhouette_score(self.correlations_mean, kmeans.labels_)
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

        seed = NiftiMasker(self.mask_source).fit()
        labels_img = seed.inverse_transform(self.kmeans.labels_+1) # +1 because labels start from 0 
        labels_img.to_filename(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_labels_K' + str(self.k) + '.nii.gz')

        print("DONE")

    def prepare_brain_maps(self):
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

        brain_maps = np.zeros((len(np.unique(self.kmeans.labels_)), self.correlations_mean.shape[1]))
        for label in np.unique(self.kmeans.labels_):
            brain_maps[label,:] = np.mean(self.correlations_mean[np.where(self.kmeans.labels_==label),:],axis=1)

        print("...Save as nifti files")
        brain_mask = NiftiMasker(self.mask_target).fit()
        for label in np.unique(self.kmeans.labels_):
            brain_map_img = brain_mask.inverse_transform(brain_maps[label,:])
            brain_map_img.to_filename(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_brain_pattern_K' + str(self.k) + '_' + str(label+1) + '.nii.gz')

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


        
