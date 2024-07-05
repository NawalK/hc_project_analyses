import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import contingency_matrix
from scipy import stats
import scipy.cluster.hierarchy as hc
import numpy as np
import itertools
from nilearn.maskers import NiftiMasker
from nilearn import plotting, surface
import nibabel as nib
import seaborn as sns
import os, json
from matplotlib.colors import ListedColormap
from compute_similarity import compute_similarity


class FC_Parcellation:
    '''
    The FC_Parcellation class is used to perform the parcellation of a specific roi
    based on the FC profiles of each of its voxels
    '''
    
    def __init__(self, config, overwrite=False):
        self.config = config # Load config info
        self.clusters = {}
       
        # Read mask data
        self.mask_source_path = self.config['main_dir']+self.config['masks']['source']
        self.mask_target_path = self.config['main_dir']+self.config['masks']['target']
        self.mask_source = nib.load(self.mask_source_path).get_data().astype(bool)
        self.mask_target = nib.load(self.mask_target_path).get_data().astype(bool)
        self.levels_masked = nib.load(self.config['main_dir'] + self.config['spinal_levels']).get_fdata()[self.mask_source] 
        self.levels_masked_vec = self.levels_masked.flatten()[self.levels_masked.flatten()>0]
        self.levels_order = np.argsort(self.levels_masked.astype(int))
        self.levels_sorted = self.levels_masked[self.levels_order]
        
        # Create folder structure and save config file as json for reference
        path_to_create = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/'
        path_config = path_to_create + 'config_' + self.config['output_tag'] + '.json'

        if os.path.exists(path_to_create) and not overwrite:
            print("An analysis with this tag has already been done - Loading configuration from existing file!")
            with open(path_config) as config_file:
                self.config = json.load(config_file)
        else:
            print("Creating instance based on provided configuration file")

            os.makedirs(os.path.dirname(path_to_create),exist_ok=True)
            for folder in ['fcs', 'source', 'target']:
                os.makedirs(os.path.join(path_to_create, folder), exist_ok=True)
            with open(path_config, 'w') as f:
                json.dump(config,f)
            
            self.config = config
            
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
        if not overwrite and os.path.isfile(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '.npy'):
            print(f"... FC already computed")
        
        else: # Otherwise we compute FC    
            print(f"... Loading data")
            data_source = nib.load(self.config['main_dir'] + self.config['smooth_dir'] + 'sub-'+ sub + '/spinalcord/sub-' + sub + self.config['file_tag']['spinalcord']).get_fdata() # Read the data as a matrix
            data_target = nib.load(self.config['main_dir'] + self.config['smooth_dir'] + 'sub-'+ sub + '/brain/sub-' + sub + self.config['file_tag']['brain']).get_fdata() 

            print(f"... Computing FC for all possibilities")
            # Create empty array
            fc = np.zeros((np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_target)))
            data_source_masked = data_source[self.mask_source]
            data_target_masked = data_target[self.mask_target] 
            
            if standardize:
                data_source_masked = stats.zscore(data_source_masked, axis=1).astype(np.float32)
                data_target_masked = stats.zscore(data_target_masked, axis=1).astype(np.float32)

            print("... Computing correlation coefficient")
            fc = self._corr2_coeff(data_source_masked,data_target_masked)
            print(f"... Fisher transforming correlations")
            # Set values slightly below 1 or above -1 (for use with, e.g., arctanh) [FROM CBP TOOLS]
            fc[fc >= 1] = np.nextafter(np.float32(1.), np.float32(-1))
            fc[fc <= -1] = np.nextafter(np.float32(-1.), np.float32(1))
            fc = fc.astype(np.float32)
            fc = np.arctanh(fc)
           
            # Also compute similarity matrix
            print("... Computing similarity matrix")
            sim = np.corrcoef(fc)
            # Save everything   
            np.save(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '.npy',fc)
            np.save(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '_sim.npy',sim)
        
        print("\n\033[1mDONE\033[0m")

    def compute_mean_fc_sim(self, overwrite=False):
        '''
        To compute mean similarity and FC across all participants
        
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
        path_mean_fc = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_mean.npy'
        if not overwrite and os.path.isfile(path_mean_fc):
            print(f"... Mean FC already computed")
        else:
            print(f"... Computing mean FC")
            fc_all = np.zeros((len(self.config['list_subjects']),np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_target)))
            for sub_id,sub_name in enumerate(self.config['list_subjects']):
                fc_all[sub_id,:,:] = np.load(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub_name + '.npy')            
            fc_mean = np.mean(fc_all, axis=0)
            np.save(path_mean_fc,fc_mean)

        # Compute mean similarity matrix 
        path_mean_sim = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_mean_sim.npy'
        if not overwrite and os.path.isfile(path_mean_sim):
            print(f"... Mean similarity matrix already computed")
        else:
            print(f"... Computing mean similarity matrix")
            sim_all = np.zeros((len(self.config['list_subjects']),np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_source)))
            for sub_id,sub_name in enumerate(self.config['list_subjects']):
                sim_all[sub_id,:,:] = np.load(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub_name + '_sim.npy')            
            sim_mean = np.mean(sim_all, axis=0)
            np.save(path_mean_sim,sim_mean)

        print("\n\033[1mDONE\033[0m")     
        
    def stability_similarity(self, overwrite=False):
        '''
        To compute the stability of similarity profiles across participants
        (i.e., correlation between profiles of each pair of participants for every spinal cord voxel)
        Voxelwise maps are averaged to obtain a single map representing the mean stability across all individuals.
        
        Inputs
        ------------
        overwrite : boolean
            if set to True, labels are overwritten (default = False)

        Output
        ------------
        stability.npy
            stability maps for each combination of participants
        stability.nii.gz
            average stability map across participants
        '''
        
        print(f"\033[1mSTABILITY ANALYSIS\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        # Path to create folder structure
        path_source = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/'
        path_stability = path_source + self.config['output_tag'] + '_stability'
        if not overwrite and os.path.isfile(path_stability + '.nii.gz'):
            print(f"\033[1mLOO maps stability already computed \033[0m")
        else:
            # Otherwise, we need to load similarity matrices
            print(f"... Loading similarity matrices")
            sim_all = np.zeros((len(self.config['list_subjects']),np.count_nonzero(self.mask_source),np.count_nonzero(self.mask_source)))
            for sub_id,sub_name in enumerate(self.config['list_subjects']):
                sim_all[sub_id,:,:] = np.load(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub_name + '_sim.npy')            

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
            # Compute average stability maps and save as nifti
            average_stability_map = np.mean(stability_maps, axis=0)
            source_mask = NiftiMasker(self.mask_source_path).fit()
            average_stability_map_img = source_mask.inverse_transform(np.squeeze(average_stability_map)) 
            average_stability_map_img.to_filename(path_stability + '.nii.gz')
            
        print("\n\033[1mDONE\033[0m") 
    
    def run_clustering(self, k_range, sub=None, take_mean=False, save_nifti=True, overwrite=False):
        '''  
        Run clustering for a range of k values

        Inputs
        ------------
        sub : str 
            subject on which to run the correlations
            Note: not considered if take_mean = True
        k_range : int, array or range
            number of clusters  
        take_mean : boolean
            if set to True, clustering is done on the mean sim values across all participants
        save_nifi : boolean
            if set to True, labels are saved as .nii.gz (default = True)
        overwrite : boolean
            if set to True, labels are overwritten (default = False)
        
        Output
        ------------
        dict_clustering_indiv/mean :  dict
            labels corresponding to the clustering for each k (e.g., dict_clustering_indiv['5'] contains labels of a particular subject for k=5)
            Note: there is one dict per subject or for the mean, saved as a .pkl file for easy access
        '''

        print(f"\033[1mCLUSTERING AT THE INDIVIDUAL LEVEL\033[0m")
        print(f"\033[37mK value(s) = {k_range}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        if take_mean:
            print(f"\033[1mClustering will be done on the mean across participants!\033[0m")

        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range

        # Path to create folder structure
        path_source = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/source/'

        sim_not_loaded = True # To avoid loading fc multiple times
        # Check if file already exists
        for k in k_range:
            print(f"K = {k}")
            
            # Create folder structure if needed
            for folder in ['indiv_labels','mean_labels']:
                os.makedirs(os.path.join(path_source, 'K'+str(k), folder), exist_ok=True)

            # Check if file already exists
            path_indiv_labels = path_source + 'K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_labels_k' + str(k) if take_mean else path_source + 'K' + str(k) + '/indiv_labels/' + self.config['output_tag'] + '_' + sub + '_labels_k' + str(k)
            
            if not overwrite and os.path.isfile(path_indiv_labels + '.npy'):
                print(f"... Labels already computed")
            
            # Otherwise, we compute them
            else:
                if sim_not_loaded is True:
                    print(f"... Loading similarity matrix from file")
                        
                    if take_mean:
                        data_to_cluster = np.load(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_mean_sim.npy')
                    else:
                        data_to_cluster = np.load(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_' + sub + '_sim.npy')
                        
                    sim_not_loaded = False # To avoid loading fc multiple times

                print(f"... Running agglomerative clustering")
                # Dict containing parameters
                agglom_kwargs = {'n_clusters': k, 'linkage': 'average', 'metric': 'precomputed'}
                agglom_clusters = AgglomerativeClustering(**agglom_kwargs)
                agglom_clusters.fit(1-data_to_cluster)
                labels = agglom_clusters.labels_
    
                np.save(path_indiv_labels + '.npy', labels.astype(int))

                if save_nifti:
                    seed = NiftiMasker(self.mask_source_path).fit()
                    labels_img = seed.inverse_transform(labels.astype(int)+1) # +1 because labels start from 0 
                    labels_img.to_filename(path_indiv_labels + '.nii.gz')

        print("\n")

    def plot_dendrogram(self, k_range, overwrite=False):
        '''  
        Plot dendrogram next to re-organize similarity matrix 

        Inputs
        ------------
        k_range : int, array or range
            number of clusters  
        overwrite : boolean
            if set to True, plots are overwritten (default = False)
        
        Output
        ------------
        dendrogram :  png
            image of the dendrogram and re-ordered similarity matrix
        '''

        print(f"\033[1mPLOT DENDROGRAM\033[0m")
        print(f"\033[37mK value(s) = {k_range}\033[0m")
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")
        
        # If only one k value is given, convert to range
        k_range = range(k_range,k_range+1) if isinstance(k_range,int) else k_range

        # Path to create folder structure
        path_source = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/source/'

        sim_not_loaded = True # To avoid loading fc multiple times
        # Check if file already exists
        for k in k_range:
            print(f"K = {k}")
            
            # Create folder structure if needed
            for folder in ['indiv_labels','indiv_labels_relabeled','group_labels','mean_labels']:
                os.makedirs(os.path.join(path_source, 'K'+str(k), folder), exist_ok=True)

            # Check if file already exists
            path_dendrogram = path_source + 'K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_labels_k' + str(k) + '_dendrogram'
            
            if not overwrite and os.path.isfile(path_dendrogram + '.png'):
                print(f"... Image already exists")
            # Otherwise, we compute them
            else:
                if sim_not_loaded is True:
                    print(f"... Loading similarity matrix from file")    
                    data_to_cluster = np.load(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/fcs/' + self.config['output_tag'] + '_mean_sim.npy')
                    sim_not_loaded = False # To avoid loading fc multiple times

                print(f"... Computing dendrogram")
                upper_triangular = data_to_cluster[np.triu_indices(data_to_cluster.shape[0], k=1)]
                DF_dism = 1-upper_triangular
                linkage = hc.linkage(DF_dism, method='average',optimal_ordering=True)       
    
                # Define colors of cluster labels
                cmap_lbl = plt.get_cmap('Greys')
                colors_lbl = [cmap_lbl(i / k) for i in range(k)]
                color_mapping_lbl = {i + 1: colors_lbl[i] for i in range(k)}
                kvalues = hc.cut_tree(linkage, n_clusters=k)
                colors_lbl = [color_mapping_lbl[value] for value in kvalues.flatten()+1]

                # Define colors of spinal levels
                num_colors_sc = 7
                colors_sc = ["#1A04A4",'#0070FF','#07F6E0', "#9BFF00",'#E8F703', '#FA9C03', '#FF3A00']
                color_mapping_sc = {i + 1: colors_sc[i] for i in range(num_colors_sc)}
                colors_sc = [color_mapping_sc[value] for value in self.levels_masked_vec]
                g=sns.clustermap(data_to_cluster, row_linkage=linkage, row_colors=[colors_lbl,colors_sc], col_linkage=linkage, col_colors=[colors_lbl,colors_sc], cmap='RdBu_r', vmin=-0.6, vmax=0.6,xticklabels=False,yticklabels=False)
                g.ax_col_dendrogram.set_visible(False)
                g.savefig(path_dendrogram + '.png',format='png')

        print("\n")        

    def subject_variability(self, k_range, overwrite=False):
        '''  
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
            path_source = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/source/K' + str(k) + '/indiv_labels/' 
            path_ref = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/source/K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_labels_k' + str(k) + '.npy'
            # Load file with metrics to similarity indiv vs group labels
            if not overwrite and os.path.isfile(path_source + self.config['output_tag'] + '_all_labels_k' + str(k) + '_relabeled.npy'):
                print(f"... Relabeling already computed")
            else: # Create an empty dataframe with the needed columns
                ref_labels = np.load(path_ref)
                relabeled_labels = np.zeros((len(self.config['list_subjects']),ref_labels.shape[0]))
                seed = NiftiMasker(self.mask_source_path).fit()
                for sub_id, sub in enumerate(self.config['list_subjects']): 
                    print(f"... Relabeling sub-" + sub)
                    sub_labels = np.load(path_source + self.config['output_tag'] + '_' + sub + '_labels_k' + str(k) + '.npy')
                    # Relabel the target labels to match the reference labels as closely as possible
                    relabeled_labels[sub_id,:] = self._relabel(ref_labels, sub_labels)
                # Save
                np.save(path_source + self.config['output_tag'] + '_all_labels_k' + str(k) + '_relabeled.npy', relabeled_labels.astype(int))
                print(f"... Compute distribution for each K")
                k_maps = np.zeros((k,ref_labels.shape[0]))
                for k_i in range(0,k):
                    for sub_id,sub in enumerate(self.config['list_subjects']): 
                        k_maps[k_i,:] = k_maps[k_i,:] + (relabeled_labels[sub_id,:]==k_i).astype(int)
                    labels_img = seed.inverse_transform(k_maps[k_i,:]) 
                    labels_img.to_filename(path_source + self.config['output_tag'] + '_all_labels_k' + str(k) + '_' + str(k_i+1) + '_relabeled.nii.gz')
                
                # Count how many voxels of each cluster each participant has
                clusters_val_per_sub = np.sum(relabeled_labels.astype(int) == np.arange(k)[:, None, None], axis=2)
                # Check how many participants have a corresponding cluster
                participants_per_cluster = np.sum(clusters_val_per_sub > 0, axis=1)
                # Save as txt file
                np.savetxt(path_source + self.config['output_tag'] + '_all_labels_k' + str(k) + '_' + str(k_i+1) + '_relabeled.txt',participants_per_cluster,fmt='%d',delimiter=',')

        print("\n\033[1mDONE\033[0m")
        
    def generate_brain_maps(self, k, order=None, overwrite=False):
        '''  
        To generate corresponding brain maps (winner take all)

        Inputs
        ------------
        k : int, array or range
            number of clusters   
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
        print(f"\033[37mOverwrite results = {overwrite}\033[0m")

        # Path to stats
        main_path = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag']
        path_mean_fc = main_path + '/fcs/' + self.config['output_tag'] + '_mean.npy'
        path_labels = main_path + '/source/K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_labels_k' + str(k) + '.npy'
 
        path_wta = main_path + '/target/K' + str(k) + '/wta_fc_mean/'
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

    def plot_brain_map(self, k, colormap=plt.cm.rainbow, save_figure=False):
        ''' Plot brain maps on inflated brain
        
        Inputs
        ------------  
        k : int
            number of clusters  
        colormap : cmap
            colormap to use, will be discretized (default = plt.cm.rainbow)
        save_figure : boolean
            Set to True to save figure (default = False)

        Outputs
        ------------
        png of the brain maps
            
        '''

        print(f"\033[1mRUN PLOTTING BRAIN MAPS\033[0m")
        print(f"\033[37mK value = {k}\033[0m")
        print(f"\033[37mSave figure = {save_figure}\033[0m")

        path_data = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/target/K' + str(k) + '/wta_fc_mean/K' + str(k) + '_wta'     
        
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
                plot_path = path_data + '_' + hemi + '.png'
                plot.savefig(plot_path)
                    
    def plot_spinal_map(self, k, colormap=plt.cm.rainbow, order=None, slice_y=None, show_spinal_levels=True, save_figure=False):
        ''' Plot spinal maps on PAM50 template (coronal views)
        
        Inputs
        ------------  
        k : int
            number of clusters  
        colormap : cmap
            colormap to use, will be discretized (default = plt.cm.rainbow)
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
        print(f"\033[37mSave figure = {save_figure}\033[0m")
  
        print("The plotting is displayed in neurological orientation (Left > Right)")

        path_data = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/source/K' + str(k) + '/mean_labels/' + self.config['output_tag'] + '_mean_labels_k' + str(k)
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

        if ~hasattr(colormap, 'colors'): # Create a discretized colormap if is not already discrete
            discretized_colormap = ListedColormap(colormap(np.linspace(0, 1, k+1)))
        elif isinstance(colormap.colors, list): # Test if colormap is discrete 
            discretized_colormap = colormap

        # Load data from images 
        img_to_show = nib.load(path_data + '.nii.gz')
        spinal_data= img_to_show.get_fdata().astype(int)

        # Re-assign numbers 
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
            np.savetxt(path_data + '_order.txt',order,fmt='%d',delimiter=',')

    def compute_similarity_spinal_levels(self, save_figure=True):
        ''' Compute the similarity of the spinal parcellation for K=7 with frostell atlas 
        
        Inputs
        ------------         
        save_figure : boolean
            Set to True to save figure (default = True)

        Outputs
        ------------
        XX_diag_sim_atlas.txt
            text file containing diagonal of similarity matrix
        XX_sim_atlas.png
            heatmap of similarity matrix
        '''     

        print(f"\033[1mCOMPUTE SIMILARITY WITH ATLAS FOR K=7\033[0m")

        path_data = self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '/source/K7/mean_labels/' + self.config['output_tag'] + '_mean_labels_k7'
        
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
        mean_dice = np.mean(np.diagonal(similarity_matrix))
        std_dice = np.std(np.diagonal(similarity_matrix))
        print(f'Mean Dice = {mean_dice:.2f}')
        print(f'Std Dice = {std_dice:.2f}')
        np.savetxt(path_data + '_diag_dice_atlas.txt',np.diagonal(similarity_matrix) ,fmt='%2f', delimiter=',')

        # Saving result and figure if applicable
        if save_figure == True:
            sns.heatmap(similarity_matrix,linewidths=.5,square=True,cmap='YlOrBr',vmin=0, vmax=1,xticklabels=orderY+1,yticklabels=np.array(range(1,8)));
            plt.savefig(path_data + '_dice_atlas.pdf', format='pdf')
                         
    def _corr2_coeff(self, arr_source, arr_target):
        # Rowwise mean of input arrays & subtract from input arrays themselves
        A_mA = arr_source - arr_source.mean(1)[:, None]
        B_mB = arr_target - arr_target.mean(1)[:, None]

        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)

        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    
    def _cluster_indices_to_adjacency(self, cluster_indices):
        
        # Create a boolean mask where True indicates elements that belong to the same cluster
        adjacency_matrix = (cluster_indices[:, np.newaxis] == cluster_indices[np.newaxis, :]).astype(int)
        
        return adjacency_matrix
    
    def _relabel(self, reference_labels, target_labels):
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