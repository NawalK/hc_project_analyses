import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import nibabel as nib

class FC_Parcellation:
    '''
    The FC_Parcellation class is used to perform the parcellation of a specific roi
    based on the FC profiles of each of its voxel
    
    Attributes
    ----------
    config : dict
    connectivity: 3D array
        connectivity matrix (nb voxels seed x nb sub x nb voxel target)
    params: dict
        parameters for the clustering
    mean_connectivity: 2D array
        mean (over subjects) connectivity matrix (nb voxels seed x nb voxel target)
    '''
    
    def __init__(self, config, connectivity, params):
        self.config = config # Load config info
        self.connectivity = connectivity
        # Take the mean connectivity over subjects
        self.mean_connectivity = np.squeeze(np.mean(connectivity,axis=1))
        self.algorithm = params.get('algorithm')
        self.init = params.get('init')
        self.max_iter = params.get('max_iter')
        self.n_init = params.get('n_init')

    def run_clustering(self,k):
        '''  
        Run k-means clustering for a specific number of clusters
        )See define_n_clusters to help pick this)
        
        Input
        ------------
        k: int
            number of clusters  

        Output
        ------------
        kmeans: KMeans
            results of the k_means 
        '''
        
        # Dict containing k means parameters
        kmeans_kwargs = {'algorithm': self.algorithm, 'init': self.init, 'max_iter': self.max_iter,
              'n_init': self.n_init}

        kmeans = KMeans(n_clusters=5,**kmeans_kwargs)
        kmeans.fit(self.mean_connectivity)

        return kmeans

    def define_n_clusters(self, k_range, plot = True):
        '''  
        Probe different number of clusters to define best option
        Two methods are used:
        - SSE for different K
        - Silhouette coefficients
        
        Inputs
        ------------
        k_range: array
            number of clusters to include in the comparison
        plot: boolean
            set to True to plot results (default = True)
         
        '''
        
        # Dict containing k means parameters
        kmeans_kwargs = {'algorithm': self.algorithm, 'init': self.init, 'max_iter': self.max_iter,
              'n_init': self.n_init}

        sse = []
        silhouette_coefficients = []

        # Compute metrics for each number of clusters
        for k in k_range:
            kmeans = KMeans(n_clusters=k,**kmeans_kwargs)
            kmeans.fit(self.mean_connectivity)
            sse.append(kmeans.inertia_)
            score = silhouette_score(self.mean_connectivity, kmeans.labels_)
            silhouette_coefficients.append(score)

            # Find knee point of SSE
            kl = KneeLocator(k_range, sse, curve="convex", direction="decreasing")
            sse_knee = kl.elbow
            
        # Two subplots
        fig, ax = plt.subplots(1, 2)
        # SSE subplot with knee highlighted
        ax[0,0].plot(k_range, sse)
        ax[0,0].xticks(k_range)
        ax[0,0].xlabel("Number of Clusters")
        ax[0,0].ylabel("SSE")
        ax[0,0].axvline(x=sse_knee, color='r')
        ax[0,0].show()
        # Silhouette coefficient
        ax[0,1].plot(k_range, silhouette_coefficients)
        ax[0,1].xticks(k_range)
        ax[0,1].xlabel("Number of Clusters")
        ax[0,1].ylabel("Silhouette Coefficient")
        ax[0,1].show()
