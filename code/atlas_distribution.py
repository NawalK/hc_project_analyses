import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd

def atlas_distribution(config, compo_type, k, thresh=2,save_results=False):
    ''' Compute the distribution of brain components into atlas regions 
        
        Inputs
        ----------
        config : dict
            Contains information regarding data savings, etc.
        compo_type : str
            Defines the types of components to explore ('icap' or 'ica')
        k : int
            Number of components
        thresh : float
            Lower threshold value to binarize components (default = 2)
        save_results : str
            Defines whether results are saved or not (default = False)
        
        Outputs
        ------------
        
        '''

    print(f"ATLAS DISTRIBUTION")

    print(f"...Opening atlas files")
    atlas_img = nib.load(config['main_dir'] + config['templates']['br_atlas'] + '.nii.gz')
    atlas = atlas_img.get_fdata()

    # Note: the atlas and label files should have the same root, just different extensions (.nii.gz vs .txt)
    #labels = nib.load(config['main_dir'] + config['templates']['br_atlas'] + '.txt')

    labels_file = open(config['main_dir'] + config['templates']['br_atlas'] + '.txt')
    labels = labels_file.read()
    labels = labels.split('\n')
    labels_file.close()
    
    print(f"...Opening component files")
    img = nib.load(config['main_dir'] + config['data'][compo_type]['br_dir']  + 'K_' + str(k) + '_' + config['data'][compo_type]['k_dir'] + config['data'][compo_type]['filename'])
    data = img.get_fdata()
    data_bin = np.where(data >= thresh, 1, 0)

    # Spatial dimensions of atlas & data should be the same
    if atlas.shape[0] == data.shape[0] and atlas.shape[1] == data.shape[1] and atlas.shape[2] == data.shape[2]:
        print(f"...Computing voxels counts")
        distribution = np.zeros((k,atlas.shape[3]))
        for compo in range(0,k):
            for region in range(0,atlas.shape[3]):
                distribution[compo,region] = np.sum(np.multiply(data_bin[:,:,:,compo], atlas[:,:,:,region]))
        distribution_df = pd.DataFrame(data = distribution, 
                  index = [f'{compo_type} {i}' for i in range(1, 16)], 
                  columns = labels)    
        plt.figure(figsize=(22, 3));
        sns.heatmap(data=distribution_df,cbar_kws={'label': '# of voxels'});
        plt.title('Distribution of components in atlas regions')
        # Saving result and figure if applicable
        if save_results == True:
            print(f"...Saving results")
            plt.savefig(config['output_dir'] + config['output_tag'] + '_atlas_distribution_' + compo_type + '.png')
            distribution_df.to_pickle(config['output_dir'] + config['output_tag'] + '_atlas_distribution_' + compo_type + '.pkl')
  
    else:
        raise(Exception(f'The dimensions of the data and atlas are not matching.'))

    print(f"DONE!")

    return distribution_df

