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
from scipy.ndimage import center_of_mass

class Plotting:
    '''
    The Plotting class is used to manipulate and visualize maps
    Attributes
    ----------
    config : dict
        '''
    
    def __init__(self, config,k, analyses,sorting_method='rostrocaudal'):
        self.config = config # load config info
        self.k=k
        self.analyses= analyses
        self.sorting_method = sorting_method
        print(self.analyses)
        
        self.data={};self.map_order={}
        for ana in self.analyses:
            self.data[ana] = nib.load(glob.glob(self.config['main_dir'] + "/" + ana +"/spinalcord_"+str(self.k)+"/Comp_zscored/"+ '*4D*.nii*')[0]).get_fdata()
            
        for ana in self.analyses:
            if ana==self.analyses[0]: # order the first dataset only
                self.map_order[ana] =self._sort_maps(self.sorting_method,ana)
                self.data[ana] = self.data[ana][:,:,:,self.map_order[ana]]
                    
        self.spinal_levels = self._match_levels()
        
    def sc_plot(self, k_per_line=None, lthresh=2.3, uthresh=4.0, centering_method='max', show_spinal_levels=False, colormap='autumn', save_results=False):
        ''' Plot components overlaid on PAM50 template (coronal and axial views are shown)
        
        Inputs
        ----------
        k_per_line: str
            Number of maps to display per line (default = will be set to total number of 4th dimension in the 4D image)
        lthresh : float
            Lower threshold value to display the maps (default = 2.3)
        uthresh : float
            Upper threshold value to display the maps (default = 4.0)
        centering_method : str
            Method to center display in the anterio-posterior direction (default = 'max')
                'max' to center based on voxel with maximum activity
                'middle' to center in the middle of the volume
        show_spinal_levels : boolean
            Defines whether spinal levels are displayed or not (default = False)
        colormap : str
            Defines colormap used to plot the maps(default = 'autumn')
        save_results : boolean
            Set to True to save figure (default = False)'''
        colormaps={};alpha={}
        if len(self.analyses)==2:
            colormaps[self.analyses[0]]='autumn'; colormaps[self.analyses[1]]='winter'
            alpha[self.analyses[0]]=1; alpha[self.analyses[1]]=0.6
            
        else:
            colormaps[self.analyses[0]]=colormap
            alpha[self.analyses[0]]=1;
        
       # Order the second dataset
        if len(self.analyses)==2:
            # calculate the dice coefficient between the two dataset
            _, _, order2 = compute_similarity(self.config, self.data[self.analyses[0]], self.data[self.analyses[1]], thresh1=lthresh, thresh2=uthresh, method='Dice', match_compo=True, plot_results=False,save_results=False)
            self.data[self.analyses[1]] = self.data[self.analyses[1]][:,:,:,order2]
               

        # By default, use a single line for all 3D maps, otherwise use provided value
        if (k_per_line is not None and k_per_line <= self.k) or k_per_line is None:
            k_per_line = self.k if k_per_line is None else k_per_line
        else:
            raise(Exception('Number of maps per line should be inferior or equal to the total number of maps.'))
        # Load template image for background
        template_img = nib.load(self.config['main_dir'] + self.config['templates']['spinalcord'])
        template_data = template_img.get_fdata()
        map_masked={}
        
        if show_spinal_levels == True: # Load levels if needed
            # Find list of spinal levels to consider (defined in config)
            levels_list = sorted(glob.glob(self.config['main_dir'] +self.config['templates']["sc_levels_path"] + 'spinal_level_*.nii.gz')) # Sorted is used to make sure files are listed from low to high number (i.e., rostro-caudally)
            levels_data = np.zeros((self.data[self.analyses[0]].shape[0],self.data[self.analyses[0]].shape[1],self.data[self.analyses[0]].shape[2],len(levels_list))) # To store spinal levels, based on size of 4D map data (corresponding to template) & number of spinal levels in template
            # Loop through levels & store data
            for lvl in range(0,len(levels_list)):
                level_img = nib.load(levels_list[lvl])
                levels_data[:,:,:,lvl] = level_img.get_fdata()
                # Mask level data to use as overlays
                levels_data = np.where(levels_data > 0, levels_data, np.nan)      
            # To mask maps, values below threshold are replaced by NaN
        for ana in self.analyses:
            map_masked[ana] = np.where(self.data[ana] > lthresh, self.data[ana], np.nan)
               
        # Compute number of columns/rows and prepare subplots accordingly 
        total_rows = (self.k//k_per_line + 1)*2 if self.k > k_per_line else 2
        fig, axs = plt.subplots(nrows=total_rows,ncols=k_per_line,figsize=(2*k_per_line, 4*total_rows))
        plt.axis('off')

        for i in range(0,self.k):
            # Column is the same for coronal & axial views
            col = i%k_per_line
            # Draw coronal views
            row_coronal = 0 if i<k_per_line else (i//k_per_line-1)*2+2
            axs[row_coronal,col].axis('off')
            axs[row_coronal,col].set_title('Comp' + str(i+1)+ '\n level ' + str(self.spinal_levels[i]+1),fontsize=18,pad=20)
            #axs[row_coronal,col].set_title(('iCAP' + str(i+1) + '\n level ' + str(self.spinal_levels[i]+1) if show_spinal_levels else 'iCAP' + str(i+1)),fontsize=18,pad=15)
            
            
            if centering_method == 'middle':
                axs[row_coronal,col].imshow(np.rot90(template_data[:,70,:]),cmap='gray',origin='lower');
                if show_spinal_levels == True:
                    axs[row_coronal,col].imshow(np.rot90(levels_data[:,70,:,self.spinal_levels[i]]),cmap='gray')
                axs[row_coronal,col].imshow(np.rot90(map_masked[ana][:,template_data.shape[1]//2,:,i]),vmin=lthresh, vmax=uthresh,cmap=colormaps[ana])
               

            elif centering_method == 'max':
                if len(self.analyses)==2:
                    overlap_map=self._overlap_maps()  
                    max_y = int(np.where(overlap_map == np.nanmax(overlap_map[:,:,:,i]))[1])
                else:
                    max_y = int(np.where(map_masked[self.analyses[0]] == np.nanmax(map_masked[self.analyses[0]][:,:,:,i]))[1])
                axs[row_coronal,col].imshow(np.rot90(template_data[:,max_y,:]),cmap='gray');
                if show_spinal_levels == True:
                    axs[row_coronal,col].imshow(np.rot90(levels_data[:,max_y,:,self.spinal_levels[i]]),cmap='gray')
                for ana in self.analyses:
                    axs[row_coronal,col].imshow(np.rot90(map_masked[ana][:,max_y,:,i]),vmin=lthresh, vmax=uthresh,cmap=colormaps[ana],alpha=alpha[ana])
              
            else:
                raise(Exception(f'{centering_method} is not a supported centering method.'))

             
            # Draw axial views
            row_axial = 1 if i<k_per_line else (i//k_per_line-1)*2+3
            axs[row_axial,col].axis('off');
            
            if len(self.analyses)==2:
                overlap_map=self._overlap_maps()  
                max_z =int(np.where(overlap_map == np.nanmax(overlap_map[:,:,:,i]))[2])
                
            else:
                max_z = int(np.where(map_masked[ana] == np.nanmax(map_masked[ana][:,:,:,i]))[2])
            axs[row_axial,col].imshow(template_data[:,:,max_z].T,cmap='gray');
            
            for ana in self.analyses:
                axs[row_axial,col].imshow(map_masked[ana][:,:,max_z,i].T,vmin=lthresh, vmax=uthresh,cmap=colormaps[ana],alpha=alpha[ana])
                     
            # To "zoom" on the spinal cord, we adapt the x and y lims
            axs[row_axial,col].set_xlim([map_masked[ana].shape[0]*0.2,map_masked[ana].shape[0]*0.8])
            axs[row_axial,col].set_ylim([map_masked[ana].shape[1]*0.2,map_masked[ana].shape[1]*0.8])
            axs[row_axial,col].set_anchor('N')

        # If option is set, save results as a png
        if save_results == True:
            if len(self.analyses)==2:
                plt.savefig(self.config['output_dir'] + 'spinalcord_k_' + str(self.k) + '_' +self.analyses[0]+'_vs_'+ self.analyses[1]+'.png')
            elif len(self.analyses)==1:
                plt.savefig(self.config['output_dir'] + 'spinalcord_k_' + str(self.k) + '_' +self.analyses[0]+'.png')
    
    
    def _sort_maps(self, sorting_method,ana):
        ''' Sort maps based on sorting_method (e.g., rostrocaudally)
        
        Inputs
        ----------
        sorting_method : str
            Method used to sort maps (e.g., 'rostrocaudal')
        Outputs
        ----------
        sort_index : list
            Contains the indices of the sorted maps   
        '''  
        if sorting_method == 'rostrocaudal':
            max_z = []
            for i in range(0,self.k):
                print(np.nanmax(self.data[ana][:,:,:,i]))
                max_z.append(int(np.where(self.data[ana] == np.nanmax(self.data[ana][:,:,:,i]))[2]))
            sort_index = np.argsort(max_z)
            sort_index= sort_index[::-1] # Invert direction to go from up to low
        elif sorting_method == 'no_sorting':
            sort_index = list(range(self.k))
        else:
            raise(Exception(f'{sorting_method} is not a supported sorting method.'))
        return sort_index

    def _match_levels(self,method="CoM"):
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
        levels_data = np.zeros((self.data[self.analyses[0]].shape[0],self.data[self.analyses[0]].shape[1],self.data[self.analyses[0]].shape[2],len(levels_list))) # To store spinal levels, based on size of 4D data (corresponding to template) & number of spinal levels in template
        spinal_levels = np.zeros(self.k,dtype='int') # To store corresponding spinal levels

        # Loop through levels & store data
        for lvl in range(0,len(levels_list)):
            level_img = nib.load(levels_list[lvl])
            levels_data[:,:,:,lvl] = level_img.get_fdata()
           
        if method=="CoM":
            
            map_masked = np.where(self.data[self.analyses[0]] >2, self.data[self.analyses[0]], 0)
            CoM = np.zeros(map_masked.shape[3],dtype='int')
            for i in range(0,self.k):
                _,_,CoM[i]=center_of_mass(map_masked[:,:,:,i])
                # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
                level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,CoM[i],:]
                
                spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)
            
            
        elif method=="max intensity":
            # For each map, find rostrocaudal position of point with maximum intensity
            max_intensity = np.zeros(self.data[self.analyses[0]].shape[3],dtype='int')
            for i in range(0,self.k):
                max_intensity[i] = np.where(self.data[self.analyses[0]] == np.nanmax(self.data[self.analyses[0]][:,:,:,i]))[2]
                #print(max_intensity)
                # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
                level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,max_intensity[i],:]
                spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)
                print(max_intensity[i]) 
        else:
            raise(Exception(f'{sorting_method} is not a supported sorting method.'))
 
        return spinal_levels
    
    def _overlap_maps(self):
        mask1=np.where(~np.isnan(self.data[self.analyses[0]]),self.data[self.analyses[1]], np.nan)
        mask2=np.where(~np.isnan(self.data[self.analyses[1]]), self.data[self.analyses[0]], np.nan)
        overlap_map=(mask1+mask2)/2
        return overlap_map
        
         