import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd

#
# NOTE 
# File structure still needs to be generalized!
#


def k_axial_distribution(config, compo_type, thresh=3, vox_percentage=70, save_results=False):
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
            Lower threshold value to binarize components (default = 3)
        vox_percentage : int
            Defines the percentage of voxels that need to be in a region to consider it matched (default = 70)
        save_results : str
            Defines whether results are saved or not (default = False)
        
        Outputs
        ------------
        axial_distribution_perc : df
            Dataframe containing the percentage of voxels falling in each category (see above description) '''

    # Check that passed method is okay
    if not compo_type in ("icap", "ica"):
        raise(Exception(f'Method {compo_type} is not a valid option.'))

    print(f'COMPUTING AXIAL DISTRIBUTION \n ––– Method: {compo_type} \n ––– Range: {config["k_range"]} \n ––– Threshold: {thresh} \n ––– % for matching: {vox_percentage}')

    print(f'...Loading data for the different spinal masks')

    # Create a dictionary containing the different template masks use to define axial locations 
    mask_names = ('L','R','V','D','LV','LD','RV','RD')
    masks_dict = {}
    for mask in mask_names:
        masks_dict[mask] = nib.load(config['main_dir'] + config['templates']['sc_axialdiv_path'] + 'PAM50_cord_' + mask + '.nii.gz').get_fdata()
    
    axial_distribution_counts = {}

    print(f'...Computing distribution for each K')
    for k_tot in config['spinalcord']['k_range']: # Loop through the different number of k
        
        # Prepare empty structure to store counts
        axial_distribution_counts[k_tot] = dict(zip(('Q','LR','DV','F'), [0,0,0,0]))
        
        print(f'......K = {k_tot}')    
        # TODO: generalize when file structure has been decided
        img = nib.load(config['main_dir'] + config['data'][compo_type]['sc_dir']  + 'K_' + str(k_tot) + '_' + config['data'][compo_type]['k_dir'] + config['data'][compo_type]['filename'])
        data = img.get_fdata()

        data_bin = np.where(data >= thresh, 1, 0)

        # Look through each component for a particular k
        for k in range(0,k_tot):
            total_voxels = np.sum(data_bin[:,:,:,k]) # Total number of voxels in this component
            perc_in_masks = {}
            for mask in mask_names:
                # Compute the number of voxels in different masks
                counts_in_masks = np.sum(np.multiply(data_bin[:,:,:,k],masks_dict[mask])) 
                # Take the percentage
                perc_in_masks[mask] = (counts_in_masks / total_voxels) * 100
            
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
    sns.heatmap(axial_distribution_perc_df,cmap='YlOrBr',cbar_kws={'label': '% of components'},pad=10);
    plt.title('Axial distribution for different K \n Method: ' + compo_type)
    plt.xlabel('K')
    
    # Saving result and figure if applicable
    if save_results == True:
        plt.savefig(config['output_dir'] + config['output_tag'] + '_axial_distribution_' + compo_type + '.png')
    
    print(f'DONE!')
    
    return axial_distribution_counts

            