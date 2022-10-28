import os.path
import os
from turtle import pd
from pyparsing import col
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob
from compute_similarity import compute_similarity
from scipy.ndimage import find_objects,center_of_mass,label
from threshold_map import Threshold_map

class Plotting:
    '''
    The Plotting class is used to manipulate and visualize maps
    Attributes
    ----------
    config : dict
    region : str
        Region of interest (e.g., 'spinalcord' or 'brain')
    params1,params2 : dict
        Parameters for the clustering
        - name: str with name of set
        - k: # of clusters
        - dataset: selected dataset (e.g., 'gva' or 'mtl')
        - analysis: analysis method (e.g., 'ica' or 'icap')
        Note: by default, param2=None, which means only one dataset is plotted
    sorting_method: str
        Method used to sort maps (e.g., 'rostrocaudal')
    data : array 
        Z-scored spatial maps
    map_order : int
        Order of maps based on sorting_method
    spinal_levels : array
        Spinal levels maching the components
    '''
    
    def __init__(self, config, region, params1, params2=None, sorting_method='rostrocaudal'):
        self.config = config # Load config info
        self.region = region
        self.k = {}
        # Keys are defined as dataset_method_K
        self.name1=params1.get('dataset')+'_'+params1.get('analysis')+'_'+str(params1.get('k'))
        self.k[self.name1] = params1.get('k')
        self.dataset = {}
        self.dataset[self.name1] = params1.get('dataset') 
        self.analysis = {}
        self.analysis[self.name1] = params1.get('analysis')
        if params2 is not None: # We will look into a second set of components (if params2=None, we just have one)
            self.name2 = params2.get('dataset')+'_'+params2.get('analysis')+'_'+str(params2.get('k'))
            self.k[self.name2] = params2.get('k')
            self.dataset[self.name2] = params2.get('dataset')
            self.analysis[self.name2] = params2.get('analysis')
        
        self.sorting_method = sorting_method
        self.data = {}
        self.map_order={}
        self.spinal_levels={}

        # Load components
        for set in self.k.keys():
            self.data[set] = nib.load(glob.glob(self.config['main_dir']+self.config['data'][self.dataset[set]][self.analysis[set]][self.region]['dir'] + '/K_' + str(self.k[set]) + '/comp_zscored/*' + self.config['data'][self.dataset[set]][self.analysis[set]][self.region]["tag_filename"] + '*')[0]).get_fdata()
            self.map_order[set] = self._sort_maps(self.sorting_method, set)
            self.data[set] = self.data[set][:,:,:,self.map_order[set]]  
            self.spinal_levels[set]  = self._match_levels(set)


    # ======= SPINAL CORD ========
    
    def sc_plot(self, k_per_line=None, lthresh=2.3, uthresh=4.0, auto_thresh=False, perc_thresh=90, centering_method='max', show_spinal_levels=False, colormap='autumn', save_results=False):
        ''' Plot components overlaid on PAM50 template (coronal and axial views are shown)
        
        Inputs
        ----------
        k_per_line: str
            Number of maps to display per line (default = will be set to total number of 4th dimension in the 4D image)
        lthresh : float
            Lower threshold value to display the maps (default = 2.3)
            ! is not taken into account if auto_thresh=True
        uthresh : float
            Upper threshold value to display the maps (default = 4.0)
        auto_thresh : boolean / perc_thresh : int
            If sets to True, lower threshold is computed automatically (default = False) based on z distribution (i.e., taking the perc_thres percentile, default = 90)
        centering_method : str
            Method to center display in the anterio-posterior direction (default = 'max')
                'max' to center based on voxel with maximum activity
                'middle' to center in the middle of the volume
        show_spinal_levels : boolean
            Defines whether spinal levels are displayed or not (default = False)
        colormap : str
            Defines colormap used to plot the maps(default = 'autumn')
            Note: if there are two datasets, colormaps are hardcoded to ease visualization and comparison 
        save_results : boolean
            Set to True to save figure (default = False)'''
        
        # Overwrite threshold if automatic thresholding is set to True
        # Now: based on first dataset only
        if auto_thresh == True:
            lthresh = Threshold_map(glob.glob(self.config['main_dir']+self.config['data'][self.dataset[set]][self.analysis[set]][self.region]['dir'] + '/K_' + str(self.k[self.name1]) + '/comp_zscored/*' + self.config['data'][self.dataset[set]][self.analysis[set]][self.region]["tag_filename"] + '*')[0],
                        mask=self.config['main_dir']+ self.config["masks"]["spinalcord"],
                        percentile=perc_thresh)

        colormaps={};alpha={}
        if len(self.k.keys())==2: # If there are two datasets to copare, we fix the color 
            colormaps[self.name1]='autumn'; colormaps[self.name2]='winter'
            alpha[self.name1]=1; alpha[self.name2]=0.6
            
        else:
            colormaps[self.name1]=colormap
            alpha[self.name1]=1;
        
        # Order the second dataset
        if len(self.k.keys())==2:
            # Calculate the dice coefficient between the two datasets
            _, _, order2 = compute_similarity(self.config, self.data[self.name1], self.data[self.name2], thresh1=lthresh, thresh2=uthresh, method='Dice', match_compo=True, plot_results=False, save_results=False)
            self.data[self.name2] = self.data[self.name2][:,:,:,order2] # We reorder the second dataset to match the first one
               
        # By default, use a single line for all 3D maps, otherwise use provided value
        if (k_per_line is not None and k_per_line <= self.k[self.name1]) or k_per_line is None:
            k_per_line = self.k[self.name1] if k_per_line is None else k_per_line
        else:
            raise(Exception('Number of maps per line should be inferior or equal to the total number of maps.'))
        
        # Load template image for background
        template_img = nib.load(self.config['main_dir'] + self.config['templates']['spinalcord'])
        template_data = template_img.get_fdata()
        map_masked={}
        
        if show_spinal_levels == True: # Load levels if needed
            # Find list of spinal levels to consider (defined in config)
            levels_list = sorted(glob.glob(self.config['main_dir'] +self.config['templates']["sc_levels_path"] + 'spinal_level_*.nii.gz')) # Sorted is used to make sure files are listed from low to high number (i.e., rostro-caudally)
            levels_data = np.zeros((self.data[self.name1].shape[0],self.data[self.name1].shape[1],self.data[self.name1].shape[2],len(levels_list))) # To store spinal levels, based on size of 4D map data (corresponding to template) & number of spinal levels in template
            # Loop through levels & store data
            for lvl in range(0,len(levels_list)):
                level_img = nib.load(levels_list[lvl])
                levels_data[:,:,:,lvl] = level_img.get_fdata()
                # Mask level data to use as overlays
                levels_data = np.where(levels_data > 0, levels_data, np.nan)      
            # To mask maps, values below threshold are replaced by NaN
        for set in self.k.keys():
            map_masked[set] = np.where(self.data[set] > lthresh, self.data[set], np.nan)
               
        # Compute number of columns/rows and prepare subplots accordingly 
        total_rows = (self.k[self.name1]//k_per_line + 1)*2 if self.k[self.name1] > k_per_line else 2
        fig, axs = plt.subplots(nrows=total_rows,ncols=k_per_line,figsize=(2*k_per_line, 4*total_rows))
        plt.axis('off')

        for i in range(0,self.k[self.name1]):
            # Column is the same for coronal & axial views
            col = i%k_per_line
            # Draw coronal views
            row_coronal = 0 if i<k_per_line else (i//k_per_line-1)*2+2
            axs[row_coronal,col].axis('off')
            axs[row_coronal,col].set_title('Comp' + str(i+1)+ '\n level ' + str(self.spinal_levels[self.name1][i]+1),fontsize=18,pad=20)
            
            if centering_method == 'middle':
                axs[row_coronal,col].imshow(np.rot90(template_data[:,70,:]),cmap='gray',origin='lower');
                if show_spinal_levels == True:
                    axs[row_coronal,col].imshow(np.rot90(levels_data[:,70,:,self.spinal_levels[self.name1][i]]),cmap='gray')
                axs[row_coronal,col].imshow(np.rot90(map_masked[self.name1][:,template_data.shape[1]//2,:,i]),vmin=lthresh, vmax=uthresh,cmap=colormaps[self.name1])
               

            elif centering_method == 'max':
                if len(self.k.keys())==2:
                    overlap_map=self._overlap_maps()  
                    max_y = int(np.where(overlap_map == np.nanmax(overlap_map[:,:,:,i]))[1])
                else:
                    max_y = int(np.where(map_masked[self.name1] == np.nanmax(map_masked[self.name1][:,:,:,i]))[1])
                axs[row_coronal,col].imshow(np.rot90(template_data[:,max_y,:]),cmap='gray');
                if show_spinal_levels == True:
                    axs[row_coronal,col].imshow(np.rot90(levels_data[:,max_y,:,self.spinal_levels[self.name1][i]]),cmap='gray')
                for set in self.k.keys():
                    axs[row_coronal,col].imshow(np.rot90(map_masked[set][:,max_y,:,i]),vmin=lthresh, vmax=uthresh,cmap=colormaps[set],alpha=alpha[set])
              
            else:
                raise(Exception(f'{centering_method} is not a supported centering method.'))

             
            # Draw axial views
            row_axial = 1 if i<k_per_line else (i//k_per_line-1)*2+3
            axs[row_axial,col].axis('off');
            
            if len(self.k.keys())==2:
                overlap_map=self._overlap_maps()  
                max_z =int(np.where(overlap_map == np.nanmax(overlap_map[:,:,:,i]))[2])
                
            else:
                max_z = int(np.where(map_masked[self.name1] == np.nanmax(map_masked[self.name1][:,:,:,i]))[2])
            axs[row_axial,col].imshow(template_data[:,:,max_z].T,cmap='gray');
            
            for set in self.k.keys():
                axs[row_axial,col].imshow(map_masked[set][:,:,max_z,i].T,vmin=lthresh, vmax=uthresh,cmap=colormaps[set],alpha=alpha[set])
                     
            # To "zoom" on the spinal cord, we adapt the x and y lims
            axs[row_axial,col].set_xlim([map_masked[self.name1].shape[0]*0.2,map_masked[self.name1].shape[0]*0.8])
            axs[row_axial,col].set_ylim([map_masked[self.name1].shape[1]*0.2,map_masked[self.name1].shape[1]*0.8])
            axs[row_axial,col].set_anchor('N')

        # If option is set, save results as a png
        #if save_results == True:
        #    if len(self.k.keys())==2:
        #        plt.savefig(self.config['output_dir'] + 'spinalcord_k_' + str(self.k) + '_' +self.analysis[0]+'_vs_'+ self.analysis[1]+ ' _thr' + str(lthresh)+ '.png')
        #    elif len(self.k.keys())==1:
        #        plt.savefig(self.config['output_dir'] + 'spinalcord_k_' + str(self.k) + '_' +self.analysis[0]+'.png')
    
    
    def _sort_maps(self, sorting_method, set):
        ''' Sort maps based on sorting_method (e.g., rostrocaudally)
        
        Inputs
        ----------
        sorting_method : str
            Method used to sort maps (e.g., 'rostrocaudal', 'no_sorting')
        Outputs
        ----------
        sort_index : list
            Contains the indices of the sorted maps   
        '''  
        if sorting_method == 'rostrocaudal':
            max_z = []
            for i in range(0,self.k[set]):
                max_z.append(int(np.where(self.data[set] == np.nanmax(self.data[set][:,:,:,i]))[2][0]))  # take the first max in z direction
                
            sort_index = np.argsort(max_z)
            sort_index= sort_index[::-1] # Invert direction to go from up to low
        elif sorting_method == 'no_sorting':
            sort_index = list(range(self.k[set]))
        else:
            raise(Exception(f'{sorting_method} is not a supported sorting method.'))
        return sort_index

    def _match_levels(self, set, method="CoM"):
        ''' Match maps to corresponding spinal levels
        Output
        ----------
        spinal_levels : list
            Array containing one value per map
                C1 = 1, C2 = 2, C3 = 3, C4 = 4, etc.
        '''
        # Find list of spinal levels to consider (defined in config)
        
        levels_list = levels_list = sorted(glob.glob(self.config['main_dir'] +self.config['templates']["sc_levels_path"] + 'spinal_level_*.nii.gz')) # Sorted is used to make sure files are listed f # Sorted is used to make sure files are listed from low to high number (i.e., rostro-caudally)
        
        # Prepare structures
        levels_data = np.zeros((self.data[set].shape[0],self.data[set].shape[1],self.data[set].shape[2],len(levels_list))) # To store spinal levels, based on size of 4D data (corresponding to template) & number of spinal levels in template
        spinal_levels = np.zeros(self.k[set],dtype='int') # To store corresponding spinal levels

        # Loop through levels & store data

        for lvl in range(0,len(levels_list)):
            level_img = nib.load(levels_list[lvl])
            levels_data[:,:,:,lvl] = level_img.get_fdata()
           
        if method=="CoM":
            map_masked = np.where(self.data[set] >2, self.data[set], 0)
            CoM = np.zeros(map_masked.shape[3],dtype='int')
            for i in range(0,self.k[set]):
                _,_,CoM[i]=center_of_mass(map_masked[:,:,:,i])
                # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
                level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,CoM[i],:]
                
                spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)
            
            
        elif method=="max intensity":
            # For each map, find rostrocaudal position of point with maximum intensity
            max_intensity = np.zeros(self.data[set].shape[3],dtype='int')
            for i in range(0,self.k[set]):
                max_intensity[i] = np.where(self.data[set] == np.nanmax(self.data[set][:,:,:,i]))[2]
                #print(max_intensity)
                # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
                level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,max_intensity[i],:]
                spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)
               
        else:
            raise(Exception(f'{method} is not a supported matching method.'))
 
        return spinal_levels
    
    def _overlap_maps(self):
        mask1=np.where(~np.isnan(self.data[self.name1]),self.data[self.name1], np.nan)
        mask2=np.where(~np.isnan(self.data[self.name2]), self.data[self.name2], np.nan)
        overlap_map=(mask1+mask2)/2
        return overlap_map
        

    # ======= BRAIN ========
        
    def brain_plot(self, k_per_line=None, lthresh=2.3, uthresh=4.0, centering_method='max', colormap='autumn', direction='yxz', save_results=False):
        
        coordinates=np.zeros((self.k,3))
        # 1. Load template image for background --------------------------------------------------------
        template_img = nib.load(self.config['main_dir'] + self.config['templates']['brain'])
        template_data = template_img.get_fdata()
        
        #2. Select & mask func data --------------------------------------------------------------------
        self.data={}
        for ana in self.analysis:
            self.data[ana] = nib.load(glob.glob(self.config['main_dir'] + "/" + ana +"/brain_"+str(self.k)+"/Comp_zscored/"+ '*4D*.nii*')[0]).get_fdata()
        map_masked= np.where(self.data[ana] > lthresh, self.data[ana], np.nan)
          
        # Compute number of columns/rows and prepare subplots accordingly 
        total_rows = (self.k//k_per_line) if self.k > k_per_line else 2
        fig, axs = plt.subplots(total_rows, k_per_line*3, figsize=(10*k_per_line, 2*total_rows),facecolor='k')
        
        for i in range(0,self.k):
            coordinates[i]=self._find_xyz_cut_coords(self.data[ana][:,:,:,i])
            
           
        for i in range(0,self.k):
            for pos,coords in enumerate(coordinates[i]):
                coords=np.int(coords)
                if direction[pos]=='x':
                    cut_bg = np.rot90(template_data[coords,:,:],3)
                    cut_func = np.rot90(map_masked[coords,:,:,i],3)
        
                if direction[pos]=='y':
                    cut_bg = np.rot90(template_data[:,coords,:],3)
                    cut_func = np.rot90(map_masked[:,coords,:,i],3)
            
        
                if direction[pos]=='z':
                    cut_bg = np.rot90(template_data[:,:,coords],3)
                    cut_func = np.rot90(map_masked[:,:,coords,i],3)
        
        
                row=0 if i<k_per_line else (i//(k_per_line))
                col=pos if i%k_per_line==0 else i%k_per_line*(3)+pos
        
                axs[row,col].imshow(cut_bg,cmap='gray',origin='lower')
                axs[row,col].imshow(cut_func,cmap=colormap,vmin=lthresh, vmax=uthresh,origin='lower')
                axs[row,col].set_aspect('equal')
                if pos==1:
                    axs[row,col].set_title("Comp " + str(i),color='w')
  
                axs[row,col].axis('off')
                axs[row,col].set_facecolor('k')

    def _find_xyz_cut_coords(self,data):
        """ Find the center of the largest cluster of activation.
        also see: https://github.com/nilearn/nilearn/blob/d53169c6af1cbb3db3485c9480a3e7cb31c2537d/nilearn/plotting/find_cuts.py"""""
        offset = np.zeros(3)
        my_map=data
        
        mask = np.asarray(np.abs(my_map) > (3))
        mask=self._largest_connected_component(mask)
        
        slice_x, slice_y, slice_z= find_objects(mask.astype(int))[0] #  Slices correspond to the minimal parallelepiped that contains the object.
        my_map = my_map [slice_x, slice_y, slice_z,]
        mask = mask[slice_x, slice_y, slice_z]
        my_map *= mask
        
        offset += [slice_x.start, slice_y.start, slice_z.start]
        
        # For the second threshold, we use a mean, as it is much faster,
        # although it is less robust
        second_threshold = np.abs(np.nanmean(my_map[mask]))
        second_mask = (np.abs(my_map) > second_threshold)
        if second_mask.sum() > 50:
            my_map*=self._largest_connected_component(second_mask)
       
        cut_coords = center_of_mass(np.abs(my_map))
        
        x_map, y_map, z_map= cut_coords + offset
        coordinates=[x_map, y_map, z_map]
        
        
            
        return coordinates
            
    def _largest_connected_component(self,volume):
        labels, label_nb = label(volume) #label_nb= number of cluster founded
        label_count = np.bincount(labels.ravel().astype(int)) # number of consecutive voxels in each cluster
        label_count[0] = 0 # ingnore the first label
        return labels == label_count.argmax() # extract the data from the max cluster


            
        

