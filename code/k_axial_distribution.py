import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib

def k_axial_distribution(config, compo_type, thresh=2, vox_percentage=75, save_results=False):
    ''' Compares the axial distribution of components for different Ks
        Categories:
            - Q: + than vox_percentage of voxels in a quadrant (e.g., LR, DV, etc.)
            - L/R: + than vox_percentage of voxels in the left/right hemicord
            - D/V: + than vox_percentage of voxels in the dorsal/ventral hemicord
            - F: else (i.e., voxels "evenly" distributed over axial plane)
      
        Inputs
        ----------
        config : dict
            Contains information regarding data savings, etc.
        compo_type : str
            Defined the types of components to explore ('icap' or 'ica')
        thresh : float
            Lower threshold value to binarize components (default = 2)
        save_results : str
            Defines whether results are saved or not (default = False)
        
        Outputs
        ------------
        axial_distribution_perc : 2D array
            Matrix containing the percentage of voxels falling in each category (see above description) '''

    # Check that passed method is okay
    if not compo_type in ("icap", "ica"):
        raise(Exception(f'Method {compo_type} is not a valid option.'))

    print(f'COMPUTING AXIAL DISTRIBUTION \n ––– Method: {compo_type} \n ––– Range: {config["k_range"]}')
    
    possib_names = ('Q','LR','DV','F')
    axial_distribution_counts = np.zeros((len(config['k_range']),len(possib_names)))

    print(f'...Loading data for the different spinal masks')
    mask_l = nib.load(config['main_dir'] + config[compo_type]['main_dir'] + 'K_' + k_tot + config[compo_type]['k_dir'] + config[compo_type]['filename']).get_fdata()
           

    for k_tot in range(0,config['k_range']): # Loop through the different number of k
        print(f'...Loading data for K = {k_tot}')
        
        # TODO: generalize when file structure has been decided
        img = nib.load(config['main_dir'] + config[compo_type]['main_dir'] + 'K_' + k_tot + config[compo_type]['k_dir'] + config[compo_type]['filename'])
        data = img.get_fdata()

        print(f'...Binarize data with threshold = {thresh}')
        data_bin = np.where(data >= thresh, 1, 0)

        # Initialize counts
        counts_q = 0
        counts_lr = 0
