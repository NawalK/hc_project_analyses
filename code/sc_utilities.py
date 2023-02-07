import glob 
import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass,label
from collections import Counter 

def sort_maps(data, sorting_method,threshold):
    ''' Sort maps based on sorting_method (e.g., rostrocaudally)
    
    Inputs
    ----------
    data : array
        4D array containing the k maps to order  
    sorting_method : str
        Method used to sort maps (e.g., 'rostrocaudal', 'rostrocaudal_CoM', 'no_sorting')
    
    Output
    ----------
    sort_index : list
        Contains the indices of the sorted maps   
    '''  
    if sorting_method == 'rostrocaudal':
        max_z = []; 
        for i in range(0,data.shape[3]):
            max_z.append(int(np.where(data == np.nanmax(data[:,:,:,i]))[2][0]))  # take the first max in z direction
                
        sort_index = np.argsort(max_z)
        sort_index= sort_index[::-1] # Invert direction to go from up to low
    
    elif sorting_method == 'rostrocaudal_CoM':
        cm_z=[]
        data_bin = np.where(data > threshold, 1, 0) # binarize data
           
        # We calculate the center of mass of the largest clusters
        for i in range(0,data.shape[3]):
            # Label data to find the different clusters
            lbl1 = label(data_bin[:,:,:,i])[0]
            cm = center_of_mass(data_bin[:,:,:,i],lbl1,Counter(lbl1.ravel()).most_common()[1][0]) # take the center of mass of the larger cluster
            cm_z.append(cm[2])
        
        sort_index = np.argsort(cm_z)
        sort_index= sort_index[::-1] # Invert direction to go from up to low
            
        
    
    elif sorting_method == 'no_sorting':
        sort_index = list(range(data.shape[3]))
    else:
        raise(Exception(f'{sorting_method} is not a supported sorting method.'))
    return sort_index

def match_levels(config, data, method="CoM"):
    ''' Match maps to corresponding spinal levels
    Inputs
    ----------
    config : dict
        Content of the configuration file
    data : array
        4D array containing the k maps to match

    Output
    ----------
    spinal_levels : list
        Array containing one value per map
            C1 = 1, C2 = 2, C3 = 3, C4 = 4, etc.
    '''
    # Find list of spinal levels to consider (defined in config)
        
    levels_list = levels_list = sorted(glob.glob(config['main_dir'] + config['templates']["sc_levels_path"] + 'spinal_level_*.nii.gz')) # Sorted is used to make sure files are listed f # Sorted is used to make sure files are listed from low to high number (i.e., rostro-caudally)
        
    # Prepare structures
    levels_data = np.zeros((data.shape[0],data.shape[1],data.shape[2],len(levels_list))) # To store spinal levels, based on size of 4D data (corresponding to template) & number of spinal levels in template
    spinal_levels = np.zeros(data.shape[3],dtype='int') # To store corresponding spinal levels

    # Loop through levels & store data
    for lvl in range(0,len(levels_list)):
        level_img = nib.load(levels_list[lvl])
        levels_data[:,:,:,lvl] = level_img.get_fdata()
           
    if method=="CoM":
        map_masked = np.where(data > 1.5, data, 0) # IMPORTANT NOTE: here, a low threshold at 1.5 is used, as the goal is to have rough maps to match to levels
        CoM = np.zeros(map_masked.shape[3],dtype='int')
        for i in range(0,data.shape[3]):
            _,_,CoM[i]=center_of_mass(map_masked[:,:,:,i])
            # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
            level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,CoM[i],:]
            
            spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)
            
    elif method=="max intensity":
        # For each map, find rostrocaudal position of point with maximum intensity
        max_intensity = np.zeros(data.shape[3],dtype='int')
        for i in range(0,data.shape[3]):
            max_intensity[i] = np.where(data == np.nanmax(data[:,:,:,i]))[2]
            #print(max_intensity)
            # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
            level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,max_intensity[i],:]
            spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)
               
    else:
        raise(Exception(f'{method} is not a supported matching method.'))
 
    return spinal_levels


            
        

