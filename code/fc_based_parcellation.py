import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from nilearn.maskers import NiftiMasker

class FC_Parcellation:
    '''
    The FC_Parcellation class is used to perform the parcellation of a specific roi
    based on the FC profiles of each of its voxel
    
    Attributes
    ----------
    config : dict
    connectivity: 3D array
        connectivity matrix (nb voxels seed x nb sub x nb voxel target)
    seed_folder: str
        folder containing seed tcs, etc (& later results) (e.g., /spinalcord_icas_k_9/)
    seed_name: str
            name of the seed (e.g., "C3")
    seed_mask: str
            path to the binary mask of the seed
    params: dict
        parameters for the clustering
    mean_connectivity: 2D array
        mean (over subjects) connectivity matrix (nb voxels seed x nb voxel target)
    '''
    
    def __init__(self, config, connectivity, seed_folder, seed_mask, seed_name, target_mask, params):
        self.config = config # Load config info
        self.connectivity = connectivity
        # Take the mean connectivity over subjects
        self.mean_connectivity = np.squeeze(np.mean(connectivity,axis=1))
        self.seed_folder = seed_folder
        self.seed_mask = seed_mask
        self.seed_name = seed_name
        self.target_mask = target_mask
        self.init = params.get('init')
        self.max_iter = params.get('max_iter')
        self.n_init = params.get('n_init')

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
