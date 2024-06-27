import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import norm, ttest_1samp
from nilearn import plotting as nlp  
import os, csv
from matplotlib.colors import ListedColormap
from neuromaps import transforms
from nilearn import surface

class TaskAnalysis:
    '''
    The TaskAnalysis class is used to perform 3rd level analyses on the results
    of https://www.nature.com/articles/s41597-022-01644-4
    Data available here: https://openneuro.org/datasets/ds004044/versions/2.0.3
    
    Code largely inspired from:
    https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    https://nilearn.github.io/dev/auto_examples/07_advanced/plot_surface_bids_analysis.html#sphx-glr-auto-examples-07-advanced-plot-surface-bids-analysis-py

    Attributes
    ----------
    config : dict
    
    '''
    
    def __init__(self, main_dir, movements):
        self.main_dir = main_dir
        self.list_movements = movements
        # Load surfaces       
        self.inflated = '/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/templates/MNI/surf/lh.inflated'
        self.sulc = '/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/templates/MNI/surf/lh.sulc'
        self.pial = '/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/templates/MNI/surf/lh.pial'
        self.smc_mask = surface.vol_to_surf('/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/seed_to_voxels/29jan_2024/2_second_level/Task/mask/iCAPs_SMC.nii.gz', self.pial, radius=0,interpolation='nearest', kind='auto', n_samples=10, mask_img=None, depth=None) 
        
        colors = ["#000000"]
        # Create a ListedColormap from the specified colors
        self.outline_colormap = ListedColormap(colors)

        # Create results folder if it doesn't exist
        os.makedirs(os.path.join(main_dir, 'group_results'), exist_ok=True)        
        for mov in self.list_movements:
             # Create movement folder if it doesn't exist
            path_movement = os.path.join(self.main_dir, 'group_results', mov)
            os.makedirs(path_movement, exist_ok=True)

        # Open the participants TSV file to create participant lists
        with open('/media/miplab-nas2/Data3/Somato/participants.tsv', 'r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')            
            self.list_subjects = [row[0] for i, row in enumerate(reader) if i > 0]
            
    def run_level3(self, movements=None):
        print(f"\033[1mRUNNING 3RD LEVEL ANALYSIS\033[0m")

        # Movements are provided as inputs or taken from config file
        movements = self.list_movements if movements is None else movements

        # If only a string has been give, convert to list with a single element
        movements = [movements] if isinstance(movements, str) else movements

        # Create results folder if it doesn't exist
        for mov in movements:
            print(f"... Movement: {mov}")
            path_movement = self.main_dir + 'group_results/' + mov + '/'
            zstats_allsub_left = []
            for sub in self.list_subjects:
                zstat_cifti = nib.load(self.main_dir + sub + '/results/ses-1_task-motor_hp200_s4_level2.feat/' + sub + '_ses-1_task-motor_level2_zstat_' + mov + '_hp200_s4.dscalar.nii')
                zstat_surf_left, _ = self._decompose_cifti(zstat_cifti)
                zstats_allsub_left.append(zstat_surf_left)
            # Compute group level stats
            _, pval_group_left = ttest_1samp(np.array(zstats_allsub_left), 0)
            # Convert to z-values
            zval_group_left = norm.isf(pval_group_left)
            np.save(path_movement + 'ses-1_task-motor_level3_zstat_' + mov + '_hp200_s4.npy', zval_group_left)
            # Convert to .gii.gz to save
            zval_group_left_gii_image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(zval_group_left,datatype='NIFTI_TYPE_FLOAT32')])
            nib.save(zval_group_left_gii_image, path_movement + 'ses-1_task-motor_level3_zstat_' + mov + '_hp200_s4.surf.gii')
            # To convert from fsLR to fsaverage (not working on the server)
            #surf_fsaverage = transforms.fslr_to_fsaverage(path_movement + 'ses-1_task-motor_level3_zstat_' + mov + '_hp200_s4.surf.gii', target_density='164k', hemi='L', method='linear')
            #nib.save(surf_fsaverage[0], path_movement + 'ses-1_task-motor_level3_zstat_' + mov + '_hp200_s4_fsaverage.surf.gii')
        
        print("\033[1mDONE\033[0m\n")

    def plot_results(self, movements=None, threshold=5, vmin=4, vmax=10, colormap='Spectral_r'):
        print(f"\033[1mPLOTTING RESULTS\033[0m")

        # Movements are provided as inputs or taken from config file
        movements = self.list_movements if movements is None else movements

        # If only a string has been give, convert to list with a single element
        movements = [movements] if isinstance(movements, str) else movements
        
        # Create results folder if it doesn't exist
        for mov in movements:
            print(f"... Movement: {mov}")
            path_movement = self.main_dir + 'group_results/' + mov + '/'
            # Plot results and save
            task_surf = surface.load_surf_data(path_movement + 'ses-1_task-motor_level3_zstat_' + mov + '_hp200_s4_fsaverage.surf.gii')
            plot = nlp.plot_surf(self.inflated, task_surf, threshold=threshold, vmin=vmin, vmax=vmax, cmap=colormap, bg_map=self.sulc, colorbar=True, darkness=0.7)            
            plot.savefig(path_movement + 'ses-1_task-motor_level3_zstat_' + mov + '_hp200_s4_thr' + str(threshold) + '.png')

    def plot_outlines(self, movements=None, threshold=5):
        print(f"\033[1mPLOTTING OUTLINES\033[0m")

        # Movements are provided as inputs or taken from config file
        movements = self.list_movements if movements is None else movements

        # If only a string has been give, convert to list with a single element
        movements = [movements] if isinstance(movements, str) else movements
        
        # Create results folder if it doesn't exist
        for mov in movements:
            print(f"... Movement: {mov}")
            path_movement = self.main_dir + 'group_results/' + mov + '/'
            # Plot results and save
            task_surf = surface.load_surf_data(path_movement + 'ses-1_task-motor_level3_zstat_' + mov + '_hp200_s4_fsaverage.surf.gii')
            # Binarize task and mask
            task_surf_bin = np.where(task_surf > threshold, 1, 0) 
            task_surf_bin_smc = task_surf_bin*self.smc_mask
            plot = nlp.plot_surf_contours(self.inflated, task_surf_bin_smc, cmap=self.outline_colormap, levels=[1], bg_map=self.sulc, darkness=0.7)
            plot.savefig(path_movement + 'ses-1_task-motor_level3_zstat_' + mov + '_hp200_s4_thr' + str(threshold) + '_outline.png')
            
    def winner_takes_all(self, movements=None, colormap=plt.cm.rainbow):
        print(f"\033[1mRUN WINNER-TAKES-ALL ANALYSIS\033[0m")

        # Movements are provided as inputs or taken from config file
        movements = self.list_movements if movements is None else movements

        maps_mov = []
        for mov in movements:
            print(f"... Movement: {mov}")
            # Extract movement data
            path_movement = self.main_dir + 'group_results/' + mov + '/'
            maps_mov.append(np.load(path_movement + 'ses-1_task-motor_level3_zstat_' + mov + '_hp200_s4.npy')) 
            
        data=np.squeeze(np.array(maps_mov))
        # Threshold
        data_thr = np.where(data >= 5, data, 0)
        max_level_indices = np.empty((data_thr.shape[1],1))
   
        # Loop through each voxel
        for i in range(0,data_thr.shape[1]):
            i_values = data_thr[:,i]  # Get the voxel values
            max_level_index = np.argmax(i_values)  # Find the level that have the max value for this column
            if i_values[max_level_index] == 0 :
                max_level_index = -1 # if the max value is 0 put -1 to the index
            max_level_indices[i] = max_level_index+1 
        
        # Plot results and save
        discretized_colormap = ListedColormap(colormap(np.linspace(0, 1, len(movements)+1)))
        plot = nlp.plot_surf_roi(self.surface, max_level_indices, vmin=1, vmax=len(movements), cmap=discretized_colormap, colorbar=True)
        plot.savefig(self.main_dir + 'group_results/ses-1_task-motor_level3_zstat_hp200_s4_WTA.png')

        print("\033[1mDONE\033[0m\n")

    def _decompose_cifti(self,img):
        data = img.get_fdata(dtype=np.float32)
        brain_models = img.header.get_axis(1)  # Assume we know this
        return (self._surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT"),
                self._surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT"))

    def _surf_data_from_cifti(self,data, axis, surf_name):
        assert isinstance(axis, nib.cifti2.BrainModelAxis)
        for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
            if name == surf_name:                                 # Just looking for a surface
                data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
                vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
                surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
                surf_data[vtx_indices] = data
                return surf_data
        raise ValueError(f"No structure named {surf_name}")