import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from nilearn.maskers import NiftiMasker
import nibabel as nib

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
    
    def __init__(self, config, struct_source, struct_target, params_kmeans={'init':'k-means++', 'n_init':100, 'max_iter':300}):
        self.config = config # Load config info
        self.struct_source = struct_source
        self.struct_target = struct_target
        
        # Load data
        self.data_source = {}
        self.data_target = {}
        for sub in self.config['list_subjects']:
            self.data_source[sub] = nib.load(config['smooth_dir'] + 'sub-'+ sub + '/' + self.struct_source + '/sub-' + sub + config['coreg_tag'][struct_source] + '.nii').get_fdata() # Read the data as a matrix
            self.data_target[sub] = nib.load(config['smooth_dir'] + 'sub-'+ sub + '/' + self.struct_target + '/sub-' + sub + config['coreg_tag'][struct_target] + '.nii').get_fdata() 

        self.init = params_kmeans.get('init')
        self.n_init = params_kmeans.get('n_init')
        self.max_iter = params_kmeans.get('max_iter')

    def voxelwise_correlation(self, mask_source, mask_target, Fisher=True):
        '''
        To compute Pearson correlation between each voxel of mask_source to all voxels of mask_target
        mask_source/target : str
            Paths of masks defining the regions to consider
        Fisher : bool
            To Fisher-transform the correlation (default = True).
        '''
        for sub in self.config['list_subjects']

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
        kmeans.fit(self.mean_connectivity)

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
        k_range: array
            number of clusters to include in the comparison
          
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
            kmeans.fit(self.mean_connectivity)
            sse.append(kmeans.inertia_)
            score = silhouette_score(self.mean_connectivity, kmeans.labels_)
            silhouette_coefficients.append(score)

        # Find knee point of SSE
        kl = KneeLocator(k_range, sse, curve="convex", direction="decreasing")
        sse_knee = kl.elbow
            
        # Two subplots
        fig, (ax1,ax2) = plt.subplots(1, 2)
        # SSE subplot with knee highlighted
        ax1.plot(k_range, sse)
        ax1.set(xticks=k_range, xlabel = "Number of Clusters", ylabel = "SSE")
        if sse_knee is not None: 
            ax1.axvline(x=sse_knee, color='r')
        # Silhouette coefficient
        ax2.plot(k_range, silhouette_coefficients)
        ax2.set(xticks=k_range, xlabel = "Number of Clusters", ylabel = "Silhouette Coefficient")

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

        seed = NiftiMasker(self.seed_mask).fit()
        labels_img = seed.inverse_transform(self.kmeans.labels_+1) # +1 because labels start from 0 
        labels_img.to_filename(self.config['main_dir'] + self.config['seed2vox_dir'] + self.seed_folder + self.seed_name + '/cbp/labels_K' + str(self.k) + '.nii.gz')

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

        brain_maps = np.zeros((len(np.unique(self.kmeans.labels_)), self.mean_connectivity.shape[1]))
        for label in np.unique(self.kmeans.labels_):
            brain_maps[label,:] = np.mean(self.mean_connectivity[np.where(self.kmeans.labels_==label),:],axis=1)

        print("...Save as nifti files")
        brain_mask = NiftiMasker(self.target_mask).fit()
        for label in np.unique(self.kmeans.labels_):
            brain_map_img = brain_mask.inverse_transform(brain_maps[label,:])
            brain_map_img.to_filename(self.config['main_dir'] + self.config['seed2vox_dir'] + self.seed_folder + self.seed_name + '/cbp/brain_pattern_K' + str(self.k) + '_' + str(label+1) + '.nii.gz')

        print("DONE")
