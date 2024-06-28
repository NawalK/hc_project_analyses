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
        self.brain_surfaces = '/media/miplab-nas2/Data3/BMPD/hc_project/templates/surf/'

        self.smc_mask = {}
        for hemi in ['left','right']:
            self.smc_mask[hemi] = surface.vol_to_surf('/media/miplab-nas2/Data3/BMPD/hc_project/analysis/masks/brain/iCAPs_z_SMC_bin.nii.gz', self.brain_surfaces + ('lh.pial' if hemi == 'left' else 'rh.pial'), radius=0,interpolation='nearest', kind='auto', n_samples=10, mask_img=None, depth=None) 
        
        self.seed_to_voxels = '/media/miplab-nas2/Data3/BMPD/hc_project/brain_spine/results/seed_to_voxels/'

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
            zstats_allsub_right = []
            for sub in self.list_subjects:
                zstat_cifti = nib.load(self.main_dir + sub + '/results/ses-1_task-motor_hp200_s4_level2.feat/' + sub + '_ses-1_task-motor_level2_cope_' + mov + '_hp200_s4.dscalar.nii')
                zstat_surf_left, zstat_surf_right = self._decompose_cifti(zstat_cifti)
                zstats_allsub_left.append(zstat_surf_left)
                zstats_allsub_right.append(zstat_surf_right)
            # Compute group level stats
            _, pval_group_left = ttest_1samp(np.array(zstats_allsub_left), 0)
            _, pval_group_right = ttest_1samp(np.array(zstats_allsub_right), 0)
            # Convert to z-values
            zval_group_left = norm.isf(pval_group_left)
            zval_group_right = norm.isf(pval_group_right)
            np.save(path_movement + 'ses-1_task-motor_level3_zstat_left_' + mov + '_hp200_s4.npy', zval_group_left)
            np.save(path_movement + 'ses-1_task-motor_level3_zstat_right_' + mov + '_hp200_s4.npy', zval_group_right)
            # Convert to .gii.gz to save
            zval_group_left_gii_image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(zval_group_left,datatype='NIFTI_TYPE_FLOAT32')])
            nib.save(zval_group_left_gii_image, path_movement + 'ses-1_task-motor_level3_zstat_left_' + mov + '_hp200_s4.surf.gii')
            zval_group_right_gii_image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(zval_group_right,datatype='NIFTI_TYPE_FLOAT32')])
            nib.save(zval_group_right_gii_image, path_movement + 'ses-1_task-motor_level3_zstat_right_' + mov + '_hp200_s4.surf.gii')

            # To convert from fsLR to fsaverage (not working on the server)
            #surf_fsaverage_left = transforms.fslr_to_fsaverage(path_movement + 'ses-1_task-motor_level3_zstat_left_' + mov + '_hp200_s4.surf.gii', target_density='164k', hemi='L', method='linear')
            #nib.save(surf_fsaverage_left[0], path_movement + 'ses-1_task-motor_level3_zstat_left_' + mov + '_hp200_s4_fsaverage.surf.gii')
            #surf_fsaverage_right = transforms.fslr_to_fsaverage(path_movement + 'ses-1_task-motor_level3_zstat_right_' + mov + '_hp200_s4.surf.gii', target_density='164k', hemi='R', method='linear')
            #nib.save(surf_fsaverage_right[0], path_movement + 'ses-1_task-motor_level3_zstat_right_' + mov + '_hp200_s4_fsaverage.surf.gii')

        print("\033[1mDONE\033[0m\n")

    def plot_results(self, movements=None, threshold=6, vmin=4, vmax=10, colormap='Spectral_r'):
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
            for hemi in ['left','right']:
                task_surf = surface.load_surf_data(path_movement + 'ses-1_task-motor_level3_zstat_' + hemi + '_' + mov + '_hp200_s4_fsaverage.surf.gii')
                plot = nlp.plot_surf(self.brain_surfaces + ('lh.inflated' if hemi == 'left' else 'rh.inflated'), task_surf, threshold=threshold, vmin=vmin, vmax=vmax, cmap=colormap, hemi=hemi, view='lateral', bg_map=self.brain_surfaces + ('lh.sulc' if hemi == 'left' else 'rh.sulc'), colorbar=True, darkness=0.7)            
                plot.savefig(path_movement + 'ses-1_task-motor_level3_zstat_' + hemi + '_' + mov + '_hp200_s4_thr' + str(threshold) + '.png')

    def plot_outlines(self, movements=None, threshold=6):
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
            for hemi in ['left','right']:
                task_surf = surface.load_surf_data(path_movement + 'ses-1_task-motor_level3_zstat_' + hemi + '_' + mov + '_hp200_s4_fsaverage.surf.gii')
                # Binarize task and mask
                task_surf_bin = np.where(task_surf > threshold, 1, 0) 
                task_surf_bin_smc = task_surf_bin*self.smc_mask[hemi]
                plot = nlp.plot_surf_contours(self.brain_surfaces + ('lh.inflated' if hemi == 'left' else 'rh.inflated'), task_surf_bin_smc, cmap=self.outline_colormap, levels=[1], hemi=hemi, view='lateral', bg_map=self.brain_surfaces + ('lh.sulc' if hemi == 'left' else 'rh.sulc'), darkness=0.7)
                plot.savefig(path_movement + 'ses-1_task-motor_level3_zstat_' + hemi + '_' + mov + '_hp200_s4_thr' + str(threshold) + '_outline.png')
                
    def compute_overlap(self, movements=None, hemispheres=['left','right'], threshold=5):
        print(f"\033[1mCOMPUTE OVERLAP WTA-TASK\033[0m")
        
        seeds = ['C3','C4','C5','C6','C7']
        # Movements are provided as inputs or taken from config file
        movements = self.list_movements if movements is None else movements

        # If only a string has been given, convert to list with a single element
        movements = [movements] if isinstance(movements, str) else movements
        hemispheres = [hemispheres] if isinstance(hemispheres, str) else hemispheres

        print(f"... Loading wta results")
        wta_maps = {}
        mov_maps = {}
        # Load wta results for each seeds + tasks
        for seed in seeds:
            wta_maps[seed] = []
            for hemi in ['left','right']:
                tmp_map = surface.vol_to_surf(self.seed_to_voxels + 'WTA_zcorr_thr0_cluster100_s0SMC_' + seed + '_cluster50.nii.gz', self.brain_surfaces + ('lh.pial' if hemi == 'left' else 'rh.pial'), radius=0,interpolation='nearest', kind='auto', n_samples=10, mask_img=None, depth=None) 
                wta_maps[seed].append(np.where(tmp_map > 0, 1, 0))

        # Grouping seeds
        wta_maps_grouped = {}
        wta_maps_grouped['C3'] = wta_maps['C3'] 
        wta_maps_grouped['C4C5'] = [x + y for x, y in zip(wta_maps['C4'], wta_maps['C5'])]
        wta_maps_grouped['C6C7'] = [x + y for x, y in zip(wta_maps['C6'], wta_maps['C7'])]

        seeds_grouped = ['C3','C4C5','C6C7']

        print(f"... Loading movement maps")
        for mov in movements:
            mov_maps[mov] = []
            for hemi in hemispheres:
                path_movement = self.main_dir + 'group_results/' + mov + '/'
                task_surf = surface.load_surf_data(path_movement + 'ses-1_task-motor_level3_zstat_' + hemi + '_' + mov + '_hp200_s4_fsaverage.surf.gii')
                # Binarize task and mask
                mov_maps[mov].append(np.where(task_surf > threshold, 1, 0))

        print(f"... Overlaps")
        print(f"...... C3 - Jaw {np.sum(np.bitwise_and(wta_maps_grouped['C3'], mov_maps['Jaw']))/np.sum(wta_maps_grouped['C3'])}")
        print(f"...... C4C5 - Upperarm {np.sum(np.bitwise_and(wta_maps_grouped['C4C5'], mov_maps['Upperarm']))/np.sum(wta_maps_grouped['C4C5'])}")
        print(f"...... C4C5 - Forearm {np.sum(np.bitwise_and(wta_maps_grouped['C4C5'], mov_maps['Forearm']))/np.sum(wta_maps_grouped['C4C5'])}")
        print(f"...... C6C7 - Wrist {np.sum(np.bitwise_and(wta_maps_grouped['C6C7'], mov_maps['Wrist']))/np.sum(wta_maps_grouped['C6C7'])}")
        print(f"...... C6C7 - Finger {np.sum(np.bitwise_and(wta_maps_grouped['C6C7'], mov_maps['Finger']))/np.sum(wta_maps_grouped['C6C7'])}")
                
        # Initialize an empty list to store the overlap percentages
        overlap_matrix = np.zeros((len(seeds_grouped),len(movements)))

        for seed_id, seed in enumerate(seeds_grouped):
            for mov_id,mov in enumerate(movements):
                # Calculate the overlap by summing the bitwise AND operation and dividing by the total number of ones in the WTA vector
                overlap_count = np.sum(np.bitwise_and(wta_maps_grouped[seed], mov_maps[mov]))
                total_count = np.sum(wta_maps_grouped[seed])
                overlap_matrix[seed_id,mov_id] = (overlap_count / total_count) * 100 if total_count > 0 else 0 
        
        # Create a heatmap plot
        plt.imshow(overlap_matrix, cmap='Spectral_r', aspect='auto')

        # Set the ticks and labels for the axes
        plt.xticks(np.arange(len(movements)), list(mov_maps.keys()), rotation=45)
        plt.yticks(np.arange(len(seeds_grouped)), list(wta_maps_grouped.keys()))

        # Add a colorbar
        plt.colorbar(label='Overlap Percentage')

    def compute_overlap2(self, movements=None, hemispheres=['left','right'], threshold=5):
        print(f"\033[1mCOMPUTE OVERLAP WTA-TASK\033[0m")
        
        seeds = ['C3','C4','C5','C6','C7']
        # Movements are provided as inputs or taken from config file
        movements = self.list_movements if movements is None else movements

        # If only a string has been given, convert to list with a single element
        movements = [movements] if isinstance(movements, str) else movements
        hemispheres = [hemispheres] if isinstance(hemispheres, str) else hemispheres

        print(f"... Loading wta results")
        wta_maps = {}
        mov_maps = {}
        # Load wta results for each seeds + tasks
        for seed in seeds:
            wta_maps[seed] = []
            for hemi in ['left','right']:
                tmp_map = surface.vol_to_surf(self.seed_to_voxels + 'WTA_zcorr_thr0_cluster100_s0SMC_' + seed + '_cluster50.nii.gz', self.brain_surfaces + ('lh.pial' if hemi == 'left' else 'rh.pial'), radius=0,interpolation='nearest', kind='auto', n_samples=10, mask_img=None, depth=None) 
                wta_maps[seed].append(np.where(tmp_map > 0, 1, 0))

        # Grouping seeds
        wta_maps_grouped = {}
        wta_maps_grouped['C3'] = wta_maps['C3'] 
        wta_maps_grouped['C4C5'] = [x + y for x, y in zip(wta_maps['C4'], wta_maps['C5'])]
        wta_maps_grouped['C6C7'] = [x + y for x, y in zip(wta_maps['C6'], wta_maps['C7'])]

        seeds_grouped = ['C3','C4C5','C6C7']

        print(f"... Loading movement maps")
        for mov in movements:
            mov_maps[mov] = []
            for hemi in hemispheres:
                path_movement = self.main_dir + 'group_results/' + mov + '/'
                task_surf = surface.load_surf_data(path_movement + 'ses-1_task-motor_level3_zstat_' + hemi + '_' + mov + '_hp200_s4_fsaverage.surf.gii')
                # Binarize task and mask
                mov_maps[mov].append(np.where(task_surf > threshold, 1, 0))
        
        # Initialize an empty list to store the overlap percentages
        overlap_matrix = np.zeros((len(seeds_grouped),len(movements)))

        for mov_id,mov in enumerate(movements):
            overlap_count = np.zeros((len(seeds_grouped)))
            for seed_id, seed in enumerate(seeds_grouped):
                # Calculate the overlap by summing the bitwise AND operation and dividing by the total number of ones in the WTA vector
                overlap_count[seed_id] = np.sum(np.bitwise_and(wta_maps_grouped[seed], mov_maps[mov]))
            for seed_id, seed in enumerate(seeds_grouped):
                overlap_matrix[seed_id,mov_id] = overlap_count[seed_id] / np.sum(overlap_count) * 100
    
        # Create a heatmap plot
        plt.imshow(overlap_matrix, cmap='Spectral_r', aspect='auto')

        # Set the ticks and labels for the axes
        plt.xticks(np.arange(len(movements)), list(mov_maps.keys()), rotation=45)
        plt.yticks(np.arange(len(seeds_grouped)), list(wta_maps_grouped.keys()))

        # Add a colorbar
        plt.colorbar(label='Overlap Percentage')


    def winner_takes_all(self, movements=None, hemispheres=['left','right'], threshold=0, colormap=plt.cm.rainbow):
        print(f"\033[1mRUN WINNER-TAKES-ALL ANALYSIS\033[0m")

        # Movements are provided as inputs or taken from config file
        movements = self.list_movements if movements is None else movements

        # Create results folder if it doesn't exist
        for hemi in hemispheres:
            print(f"... Hemisphere: {hemi}")
            maps_mov = []
            for mov in movements:
                path_movement = self.main_dir + 'group_results/' + mov + '/'
                # Plot results and save
                task_surf = surface.load_surf_data(path_movement + 'ses-1_task-motor_level3_zstat_' + hemi + '_' + mov + '_hp200_s4_fsaverage.surf.gii')
                maps_mov.append(np.where(task_surf > threshold, task_surf, 0))
                
            data=np.squeeze(np.array(maps_mov))
            max_level_indices = np.empty((data.shape[1],1))
    
            print(f"... Computing WTA")
            # Loop through each voxel
            for i in range(0,data.shape[1]):
                i_values = data[:,i]  # Get the voxel values
                max_level_index = np.argmax(i_values)  # Find the level that have the max value for this column
                if i_values[max_level_index] == 0 :
                    max_level_index = -1 # if the max value is 0 put -1 to the index
                max_level_indices[i] = max_level_index+1 
            
            # Mask with SMC
            mask = self.smc_mask[hemi].astype(bool)
            max_level_indices[~mask] = 0

            # Plot results and save
            discretized_colormap = ListedColormap(colormap(np.linspace(0, 1, len(movements)+1)))

            print(f"... Saving")
            plot = nlp.plot_surf_roi(self.brain_surfaces +  ('lh.inflated' if hemi == 'left' else 'rh.inflated'), max_level_indices, vmin=1, vmax=len(movements), cmap=discretized_colormap, hemi=hemi, view='lateral', bg_map=self.brain_surfaces +  ('lh.sulc' if hemi == 'left' else 'rh.sulc'), colorbar=True, darkness=0.7)
            plot.savefig(self.main_dir + 'group_results/ses-1_task-motor_level3_zstat_' + hemi + '_hp200_s4_fsaverage_WTA.png')
            # Convert to .gii.gz to save
            wta_map_gii_image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(max_level_indices,datatype='NIFTI_TYPE_FLOAT32')])
            nib.save(wta_map_gii_image, self.main_dir + 'group_results/ses-1_task-motor_level3_zstat_' + hemi + '_hp200_s4_fsaverage_WTA.surf.gii')
            
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