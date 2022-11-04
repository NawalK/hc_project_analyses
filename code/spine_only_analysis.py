import nibabel as nib
import glob
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from compute_similarity import compute_similarity
from sc_utilities import sort_maps
from statistics import mean

class SpineOnlyAnalysis:
    '''
    The SpineOnlyAnalysis class is used to investigate components extracted in the spinal cord using different methods and/or datasets

    Attributes
    ----------
    config : dict
    params1,params2 : dict
        Parameters for the clustering
        - k_range: # of clusters to consider
        - dataset: selected dataset (e.g., 'gva' or 'mtl')
        - analysis: analysis method (e.g., 'ica' or 'icap')
    name1,name2: str
        Names to identify the sets (built as dataset+analysis{})    
    '''
    
    def __init__(self, config, params1, params2):
        self.config = config # Load config info

        # Define names for the two sets of interest
        self.name1=params1.get('dataset')+'_'+params1.get('analysis')
        self.name2=params2.get('dataset')+'_'+params2.get('analysis')

        # Define k range, dataset and analysis from given parameters
        self.k_range = {}
        self.k_range[self.name1] = params1.get('k_range')
        self.k_range[self.name2] = params2.get('k_range')
        self.dataset = {}
        self.dataset[self.name1] = params1.get('dataset') 
        self.dataset[self.name2] = params2.get('dataset')
        self.analysis = {}
        self.analysis[self.name1] = params1.get('analysis')
        self.analysis[self.name2] = params2.get('analysis')
        
        self.data = {} # To store the data with their initial order (i.e., as in the related nifti files)

        # Load components
        for set in self.k_range.keys(): # For each set
            self.data[set] = {}
            for k_ind,k in enumerate(self.k_range[set]): # For each k
                self.data[set][k] = nib.load(glob.glob(self.config['main_dir']+self.config['data'][self.dataset[set]][self.analysis[set]]['spinalcord']['dir'] + '/K_' + str(self.k_range[set][k_ind]) + '/comp_zscored/*' + self.config['data'][self.dataset[set]][self.analysis[set]]['spinalcord']["tag_filename"] + '*')[0]).get_fdata()
            
    def spatial_similarity(self, k1 = None, k2 = None, k_range = None, similarity_method = 'Dice', sorting_method = 'rostrocaudal', verbose=True):
        '''
        Compares spatial similarity for different sets of components.
        Can be used for different purposes:
        1 – To obtain a similarity matrix for a particular K per condition
        2 – To look at the evolution of the mean similarity across different Ks

        If single K values are specified => Method 1 is used
        If a range is given => Method 2 is used

        Inputs
        ----------
        k1, k2 : int
            K values of interest (default = None) => For method 1
        k_range : array
            Range K values of interest (default = None) => For method 2
        similarity_method : str
            Method to compute similarity (default = 'Dice')
                'Dice' to compute Dice coefficients (2*|intersection| / nb_el1 + nb_el2)
                'Euclidean distance' to compute the distance between the centers of mass
                'Cosine' to compute cosine similarity 
        sorting_method : str
            Method used to sort maps (default = 'rostrocaudal')
            Note: only used for method 1
        verbose : bool
            If True, print progress for each K (default=True)
        
        Outputs
        ----------
        XXXX to do 

        '''

        # Check if k values are provided & choose method accordingly
        if k_range == None and k1 == None and k2 == None: 
            raise(Exception(f'Either k_range or k1/k2 needs to be specified!'))
        elif k_range != None and (k1 != None or k2 != None):
            raise(Exception(f'k_range *or* k1/k2 should be specified, not both!'))
        elif k_range == None and k1 != None:
            method = 1
            if k2 == None: # If just one k is provided, we assume the same should be taken for other set
                k2 = k1
        elif k_range != None and k1 == None and k2 == None:
            method = 2
        
        # For method 1, we focus on one similarity matrix
        if method == 1:
            print(f'METHOD 1: Comparing two sets of components at specific K values \n{self.name1} at K = {k1} vs {self.name2} at K = {k2} \n')
            map_order = sort_maps(self.data[self.name1][k1], sorting_method=sorting_method) # The 1st dataset is sorted
            data_sorted = self.data[self.name1][k1][:,:,:,map_order]
            
            if similarity_method == 'Cosine': # We need masks
                mask1 = nib.load(self.config['main_dir']+self.config['masks'][self.dataset[self.name1]]['spinalcord']).get_fdata()
                mask2 = nib.load(self.config['main_dir']+self.config['masks'][self.dataset[self.name2]]['spinalcord']).get_fdata()
                similarity_matrix,_, orderY = compute_similarity(self.config, data_sorted, self.data[self.name2][k2], mask1=mask1, mask2=mask2, thresh1=2, thresh2=2, method=similarity_method, match_compo=True, verbose=False)
            else:
                similarity_matrix,_, orderY = compute_similarity(self.config, data_sorted, self.data[self.name2][k2], thresh1=2, thresh2=2, method=similarity_method, match_compo=True, verbose=False)
            plt.figure(figsize=(7,7))
            sns.heatmap(similarity_matrix, linewidths=.5, square=True, cmap='YlOrBr', vmin=0, vmax=1, xticklabels=orderY+1, yticklabels=np.array(range(1,k1+1)),cbar_kws={'shrink' : 0.8, 'label': similarity_method});
            plt.xlabel(self.name2)
            plt.ylabel(self.name1)
            
            mean_similarity = mean(x for x in np.diagonal(similarity_matrix) if x !=-1) # If ks are different, we avoid taking -1 values (no correspondance)
            print(f'The mean similarity is {mean_similarity:.2f}')
        elif method == 2:
            print('METHOD 2: Comparing two sets of components across K values')
            mean_similarity = np.empty(len(k_range), dtype=object)
            for k_ind, k in enumerate(k_range):
                if verbose == True:
                    print(f'... Computing similarity for K={k}')
                if similarity_method == 'Cosine': # We need masks
                    mask1 = nib.load(self.config['main_dir']+self.config['masks'][self.dataset[self.name1]]['spinalcord']).get_fdata()
                    mask2 = nib.load(self.config['main_dir']+self.config['masks'][self.dataset[self.name2]]['spinalcord']).get_fdata()
                    similarity_matrix,_,_ = compute_similarity(self.config, self.data[self.name1][k], self.data[self.name2][k], mask1=mask1, mask2=mask2, thresh1=2, thresh2=2, method=similarity_method, match_compo=True, verbose=False)
                else:
                    similarity_matrix,_,_ = compute_similarity(self.config, self.data[self.name1][k], self.data[self.name2][k], thresh1=2, thresh2=2, method=similarity_method, match_compo=True, verbose=False)
                mean_similarity[k_ind] = np.mean(np.diagonal(similarity_matrix)) 
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(range(1,len(k_range)+1), mean_similarity, linewidth=2, markersize=10, marker='.')
            ax.set_xticks(range(1,len(k_range)+1))
            ax.set_xticklabels(k_range)
            plt.title('Spatial similarity for different granularity levels'); plt.xlabel('K value'); plt.ylabel('Mean similarity');

        else: 
            raise(Exception(f'Something went wrong! No method was assigned...'))


    def k_axial_distribution(self, data_name, k_range=None, thresh=2, vox_percentage=70, save_results=False, verbose=True):
        '''
        Compares the axial distribution of components for different Ks
        Categories:
            - Q: + than vox_percentage of voxels in a quadrant (e.g., LR, DV, etc.)
            - L/R: + than vox_percentage of voxels in the left/right hemicord
            - D/V: + than vox_percentage of voxels in the dorsal/ventral hemicord
            - F: else (i.e., voxels "evenly" distributed over axial plane)
        
        Inputs
        ----------
        data_name : str
            Name of the set of components to analyze
        k_range : array
            Range of k values to considered (default = the one set in class attributes for the dataset defined using 'data')
        thresh : float
            Lower threshold value to binarize components (default = 2)
        vox_percentage : int
            Defines the percentage of voxels that need to be in a region to consider it matched (default = 70)
        save_results : str
            Defines whether results are saved or not (default = False)
        verbose : bool
            If True, print progress for each K (default=True)
            
        Outputs
        ------------
        axial_distribution_perc : df
            Dataframe containing the percentage of voxels falling in each category (see above description) '''

        # Set range of K
        k_range = self.k_range[data_name] if k_range == None else k_range

        print(f'COMPUTING AXIAL DISTRIBUTION \n ––– Set: {data_name} \n ––– Range: {k_range} \n ––– Threshold: {thresh} \n ––– % for matching: {vox_percentage}')

        print(f'...Loading data for the different spinal masks')

        # Create a dictionary containing the different template masks use to define axial locations 
        mask_names = ('L','R','V','D','LV','LD','RV','RD')
        masks_dict = {}
        for mask in mask_names:
            masks_dict[mask] = nib.load(self.config['main_dir'] + self.config['templates']['sc_axialdiv_path'] + 'PAM50_cord_' + mask + '.nii.gz').get_fdata()
        
        axial_distribution_counts = {}

        print(f'...Computing distribution for each K')
        for k_tot in k_range: # Loop through the different number of k
            
            # Prepare empty structure to store counts
            axial_distribution_counts[k_tot] = dict(zip(('Q','LR','DV','F'), [0,0,0,0]))
            
            if verbose == True:
                print(f'......K = {k_tot}')    

            data = self.data[data_name][k_tot]

            data_bin = np.where(data >= thresh, 1, 0)

            # Look through each component for a particular k
            for k in range(0,k_tot):
                total_voxels = np.sum(data_bin[:,:,:,k]) # Total number of voxels in this component
                perc_in_masks = {}
                for mask in mask_names:
                    # Compute the number of voxels in different masks
                    counts_in_masks = np.sum(np.multiply(data_bin[:,:,:,k],masks_dict[mask])) 
                    # Take the percentage
                    if total_voxels != 0:
                        perc_in_masks[mask] = (counts_in_masks / total_voxels) * 100
                    else:
                        perc_in_masks[mask] = 0 # To avoid having error messages if the component is empty

                # Assess assignment of component
                if perc_in_masks['LV'] >= vox_percentage or perc_in_masks['LD'] >= vox_percentage or perc_in_masks['RV'] >= vox_percentage or perc_in_masks['RD'] >= vox_percentage:
                    # If in a quadrant
                    axial_distribution_counts[k_tot]['Q'] = axial_distribution_counts[k_tot]['Q'] + 1
                elif perc_in_masks['D'] >= vox_percentage or perc_in_masks['V'] >= vox_percentage:
                    # If in the dorsal or ventral hemicord
                    axial_distribution_counts[k_tot]['DV'] = axial_distribution_counts[k_tot]['DV'] + 1
                elif perc_in_masks['L'] >= vox_percentage or perc_in_masks['R'] >= vox_percentage:
                    # If in the left or right hemicord
                    axial_distribution_counts[k_tot]['LR'] = axial_distribution_counts[k_tot]['LR'] + 1
                else:
                    # If not segregated in a particular region
                    axial_distribution_counts[k_tot]['F'] = axial_distribution_counts[k_tot]['F'] + 1

        axial_distribution_counts_df = pd.DataFrame.from_dict(axial_distribution_counts)
        # Put count as a percentage to account for different # of components
        axial_distribution_perc_df = axial_distribution_counts_df.div( axial_distribution_counts_df.sum(axis=0), axis=1).mul(100) 

        # Plot as a heatmap
        sns.heatmap(axial_distribution_perc_df,cmap='YlOrBr',cbar_kws={'label': '% of components'});
        plt.title('Axial distribution for different K \n Set: ' + data_name)
        plt.xlabel('K')
        
        # Saving result and figure if applicable
        if save_results == True:
            plt.savefig(self.config['output_dir'] + self.config['output_tag'] + '_axial_distribution_' + data_name + '.png')
        
        print(f'DONE!')
        
        return axial_distribution_counts

                