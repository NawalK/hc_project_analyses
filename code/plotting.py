import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
from compute_similarity import compute_similarity
from scipy.ndimage import find_objects,center_of_mass,label
from threshold_map import Threshold_map
from sc_utilities import match_levels, sort_maps
from skimage import measure

from nilearn import plotting
from nilearn import surface

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
        - k: # of clusters
        - dataset: selected dataset (e.g., 'gva' or 'mtl')
        - analysis: analysis method (e.g., 'ica', 'icap', 'ica_duration', 'icap_duration')
        - duration: if analysis is ica/p_duration, you need to specify duration of interest (e.g., '1min')
        - subject: to plot only a specific subject of interest (e.g., 'sub-01')
        Note: by default, param2=None, which means only one dataset is plotted
    name1, name2: str
        Names to identify the sets (built as dataset+analysis+k)
    sorting_method: str
        Method used to sort maps (e.g., 'rostrocaudal', 'rostrocaudal_CoM')
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
        self.duration = {}
        self.duration[self.name1] = params1.get('duration') # precise the duration analysis you want to plot
        self.subject = {}
        self.subject[self.name1] = params1.get('subject') # precise the subject you want to plot
        self.lthresh = {}
        self.lthresh[self.name1] = params1.get('lthresh') # precise a threshold for cluster selection (sort_map)
        
        if params2 is not None: # We will look into a second set of components (if params2=None, we just have one)
            self.name2 = params2.get('dataset')+'_'+params2.get('analysis')+'_'+str(params2.get('k'))
            if self.name1==self.name2:
                self.name2=self.name2 + "2"
            self.k[self.name2] = params2.get('k')
            self.dataset[self.name2] = params2.get('dataset')
            self.analysis[self.name2] = params2.get('analysis')
            self.duration[self.name2] = params2.get('duration')
            self.subject[self.name2] = params2.get('subject')
            self.lthresh[self.name2]  = params2.get('lthresh') # precise a threshold for cluster selection (sort_map)
        
        self.sorting_method = sorting_method
        if self.sorting_method == "rostrocaudal_CoM" and  self.lthresh[self.name1] ==None:
            raise(Exception(f'"You should predefine a threshold for sorting method rostrocaudal_CoM'))
            
            
        self.data = {} # To store the data with their initial order (i.e., as in the related nifti files)
        self.data_sorted = {} # To store the data sorted WITHIN dataset (e.g., rostrocaudally)
        self.data_matched = {} # To store the date then matched BETWEEN datasets

        self.map_order={}
        self.spinal_levels={}
        self.spinal_levels_sorted={}
        self.spinal_levels_matched={}
         
        # Load components
        for set in self.k.keys():
            if self.analysis[set]=="icap_duration" or self.analysis[set]=="ica_duration" and self.duration[set] != None and self.subject[set] == None:
                #print(self.config['main_dir']+self.config['data'][self.dataset[set]][self.analysis[set]][self.region]['dir'] + self.duration[set]  + '/K_' + str(self.k[set]) + '/comp_zscored/*' + self.config['data'][self.dataset[set]][self.analysis[set]][self.region]["tag_filename"] + '*')
                self.data[set] =nib.load(glob.glob(self.config['main_dir']+self.config['data'][self.dataset[set]][self.analysis[set]][self.region]['dir'] + self.duration[set]  + '/K_' + str(self.k[set]) + '/comp_zscored/*' + self.config['data'][self.dataset[set]][self.analysis[set]][self.region]["tag_filename"] + '*')[0]).get_fdata()
            elif self.duration[set] == None and self.subject[set] != None and self.analysis[set]!=("icap_duration" or "ica_duration"):
                #print(self.config['main_dir']+self.config['data'][self.dataset[set]][self.analysis[set]][self.region]['dir'] + '/K_' + str(self.k[set]) + '/comp_indiv/*' + self.subject[set] + '*' + self.config['data'][self.dataset[set]][self.analysis[set]][self.region]["tag_filename"] + '*')
                self.data[set] = nib.load(glob.glob(self.config['main_dir']+self.config['data'][self.dataset[set]][self.analysis[set]][self.region]['dir'] + '/K_' + str(self.k[set]) + '/comp_indiv/*' + self.subject[set] + '*' + self.config['data'][self.dataset[set]][self.analysis[set]][self.region]["tag_filename"] + '*')[0]).get_fdata()
            elif self.duration[set] == None and self.subject[set] == None and self.analysis[set]!=("icap_duration" or "ica_duration"):
                #print(self.config['main_dir']+self.config['data'][self.dataset[set]][self.analysis[set]][self.region]['dir'] + '/K_' + str(self.k[set]) + '/comp_zscored/*' + self.config['data'][self.dataset[set]][self.analysis[set]][self.region]["tag_filename"] + '*')

                self.data[set] = nib.load(glob.glob(self.config['main_dir']+self.config['data'][self.dataset[set]][self.analysis[set]][self.region]['dir'] + '/K_' + str(self.k[set]) + '/comp_zscored/*' + self.config['data'][self.dataset[set]][self.analysis[set]][self.region]["tag_filename"] + '*')[0]).get_fdata()
            else:
                raise(Exception(f'"You should define subject *or* duration, or none. If duration is chose, please choose "ica_duration" or "icap_duration" as analysis.'))
            
            self.map_order[set] = sort_maps(self.data[set], self.sorting_method, self.lthresh[self.name1])
            self.data_sorted[set] = self.data[set][:,:,:,self.map_order[set]]  
            self.spinal_levels[set] = match_levels(self.config, self.data[set],method="max intensity")
            self.spinal_levels_sorted[set] = self.spinal_levels[set][self.map_order[set]]

    # ======= SPINAL CORD ========
    
    def sc_plot(self, k_per_line=None, lthresh=None, uthresh=4.0, perc_thresh=90, template=None, centering_method='max', similarity_method='Dice', plot_mip=False, plot_overlap=True, show_spinal_levels=False, colormap_one='autumn', colormap_two=['autumn','winter'], save_results=False):
        ''' Plot components overlaid on PAM50 template (coronal and axial views are shown)
        
        Inputs
        ----------
        k_per_line : str
            Number of maps to display per line (default = will be set to total number of 4th dimension in the 4D image)
        lthresh : float
            Lower threshold value to display the maps 
        uthresh : float
            Upper threshold value to display the maps (default = 4.0)
        template : str
            To change the background if needed (default = None)
            If None, the template image defined in the config file is used
        centering_method : str
            Method to center display in the anterio-posterior direction (default = 'max')
                'max' to center based on voxel with maximum activity
                'middle' to center in the middle of the volume
        similarity_method : str
            Method to compute similarity if there are two datasets (default = Dice)
        plot_mip : boolean
            If set to True, we plot the Maximum Intensity Projection (default = False)
            ! Only possible for one dataset and centering at middle
        plot_overlap : boolean
            If set to True, we plot the outline of the overlap of the two components 
            ! Only possible if two datasets 
        show_spinal_levels : boolean
            Defines whether spinal levels are displayed or not (default = False)
        colormap_one : str
            Defines colormap used to plot the maps if one dataset (default = 'autumn')
        colormap_two : list
            Defines colormap used to plot the maps if two datasets (default = ['autumn','winter'])
        save_results : boolean
            Set to True to save figure (default = False)'''
        
        print("The plotting will be displayed in neurological orientation (Left > Right)")

        lthresh=self.lthresh[self.name1] if lthresh == None else lthresh
            
        colormaps={};alpha={}

        if plot_mip and len(self.k.keys())==2: 
            raise(Exception('Only one dataset can be plotted using the maximum intensity projection'))
        if plot_mip and centering_method != "middle":
            print("When using the maximum intensity projection, centering method is set to 'middle'")
            centering_method = "middle" # "force" centering method
        if plot_overlap and len(self.k.keys())!=2:
            raise(Exception('We need two datasets to plot the overlap'))

        # Define main and secondary dataset (useful for plotting, matching, etc.) and assign colormaps
        if len(self.k.keys())==2: 
            if self.k[self.name1] != self.k[self.name2]: # If there are two datasets with different sizes
                main_dataset = max(self.k,key=self.k.get) # We identify if a dataset is longer than the other and define it as the main dataset
                secondary_dataset = min(self.k,key=self.k.get) # Same for smaller
            elif self.k[self.name1] == self.k[self.name2]: # If there are two datasets with the same size
                main_dataset = self.name1 # We simply assign them based on the order given for the parameters
                secondary_dataset = self.name2
            colormaps[main_dataset]=colormap_two[0]; colormaps[secondary_dataset]=colormap_two[1]
            alpha[main_dataset] = 1; alpha[secondary_dataset] = 0.8     
        else:
            main_dataset = self.name1 # If only one dataset, it is for sure the longest :) 
            colormaps[main_dataset ]=colormap_one
            alpha[main_dataset] = 0.7
        
        # Order the second dataset
        if len(self.k.keys())==2:
            # Compute similarity between datasets & match them            
            _ , _, order2 = compute_similarity(self.config, self.data_sorted[secondary_dataset], self.data_sorted[main_dataset], thresh1=lthresh, thresh2=lthresh, method=similarity_method, match_compo=True, plot_results=False, save_results=False)
            self.data_matched[main_dataset] = self.data_sorted[main_dataset][:,:,:,order2] # We reorder the dataset based on similarity
            self.spinal_levels_matched[main_dataset] = self.spinal_levels_sorted[main_dataset][order2] # Also reorder spinal levels accordingly (just in case the two datasets have different sizes)
        max_k = self.k[main_dataset]
        # By default, use a single line for all 3D maps, otherwise use provided value
        if (k_per_line is not None and k_per_line <= max_k) or k_per_line is None:
            k_per_line = max_k if k_per_line is None else k_per_line
        else:
            raise(Exception('Number of maps per line should be inferior or equal to the total number of maps.'))
        
        # Load template image for background
        template_img = nib.load(self.config['main_dir'] + self.config['templates']['spinalcord']) if template is None else nib.load(template)
        template_data = template_img.get_fdata()
        map_masked = {}
        
        if show_spinal_levels == True: # Load levels if needed
            # Find list of spinal levels to consider (defined in config)
            levels_list = sorted(glob.glob(self.config['main_dir'] +self.config['templates']["sc_levels_path"] + 'spinal_level_*.nii.gz')) # Sorted is used to make sure files are listed from low to high number (i.e., rostro-caudally)
            
            levels_data = np.zeros((self.data[main_dataset].shape[0],self.data[main_dataset].shape[1],self.data[main_dataset].shape[2],len(levels_list))) # To store spinal levels, based on size of 4D map data (corresponding to template) & number of spinal levels in template
            
            # Loop through levels & store data
            for lvl in range(0,len(levels_list)):
                level_img = nib.load(levels_list[lvl])
                levels_data[:,:,:,lvl] = level_img.get_fdata()
                # Mask level data to use as overlays
                levels_data = np.where(levels_data > 0, levels_data, np.nan)      
        
        # To mask maps, values below threshold are replaced by NaN
        if len(self.k.keys())==2:
            map_masked[main_dataset] = np.where(self.data_matched[main_dataset] > lthresh, self.data_matched[main_dataset], np.nan)
            map_masked[secondary_dataset] = np.where(self.data_sorted[secondary_dataset] > lthresh, self.data_sorted[secondary_dataset], np.nan) # For the second one, we just take the sorted dataset
        else:
            map_masked[main_dataset] = np.where(self.data_sorted[main_dataset] > lthresh, self.data_sorted[main_dataset], np.nan) # Same if only one, no need to take the matched version
        # Compute number of columns/rows and prepare subplots accordingly 
        total_rows = int(np.ceil(self.k[main_dataset]/k_per_line)*2) if self.k[main_dataset] > k_per_line else 2
        _, axs = plt.subplots(nrows=total_rows,ncols=k_per_line,figsize=(2*k_per_line, 4*total_rows))
        plt.axis('off')

        for i in range(0,max_k):
            # Column is the same for coronal & axial views
            col = i%k_per_line
            # Draw coronal views
            row_coronal = 0 if i<k_per_line else (i//k_per_line-1)*2+2
            axs[row_coronal,col].axis('off')
            
            if len(self.k.keys())==2: 
                if i<self.k[secondary_dataset]:
                    axs[row_coronal,col].set_title('Main #' + str(self.map_order[main_dataset][order2[i]]+1) + '\n Sec. #' + str(self.map_order[secondary_dataset][i]+1),fontsize=18,pad=20)# + '\n Level ' + str(self.spinal_levels_sorted[secondary_dataset][i]+1),fontsize=18,pad=20)
            
            if centering_method == 'max':
                if len(self.k.keys())==2 and i<self.k[secondary_dataset]: # If maps present in both, define max based on the overlap
                    overlap_map=self._overlap_maps(i,main_dataset,secondary_dataset)  
                    max_y = int(np.where(overlap_map == np.nanmax(overlap_map))[1])
               
                else: # Otherwise, pick only based on the main dataset
                    max_size=np.where(map_masked[main_dataset] == np.nanmax(map_masked[main_dataset][:,:,:,i]))[1].size
                    if max_size>1:
                        max_y = int(np.where(map_masked[main_dataset] == np.nanmax(map_masked[main_dataset][:,:,:,i]))[1][0]) # take the first max if there are mainy
                    else:
                        max_y = int(np.where(map_masked[main_dataset] == np.nanmax(map_masked[main_dataset][:,:,:,i]))[1])
            elif centering_method == 'middle':
                max_y = template_data.shape[1]//2
                max_y = 26
            else:
                raise(Exception(f'"{centering_method}" is not a supported centering method.'))
                    
            # Show template as background
            axs[row_coronal,col].imshow(np.rot90(template_data[:,max_y,:].T,2),cmap='gray');
                
            # Show spinal levels
            if show_spinal_levels == True:
                if len(self.k.keys())==2:
                    if i<self.k[secondary_dataset]:
                        axs[row_coronal,col].imshow(np.rot90(levels_data[:,max_y,:,self.spinal_levels_sorted[secondary_dataset][i]]),cmap='gray_r')
                    else:
                        axs[row_coronal,col].imshow(np.rot90(levels_data[:,max_y,:,self.spinal_levels_matched[main_dataset][i]]),cmap='gray_r')                    
                else:
                    axs[row_coronal,col].imshow(np.rot90(levels_data[:,max_y,:,self.spinal_levels_sorted[main_dataset][i]]),cmap='gray_r',alpha=0.8)
            # Show components
            if plot_mip:
                # Compute projection
                mip = np.nansum(map_masked[main_dataset][:,:,:,i].T,axis=1)
                # Threshold
                mip = np.where(mip > lthresh, mip, np.nan)
                axs[row_coronal,col].imshow(np.rot90(mip,2),vmin=lthresh, vmax=uthresh,cmap=colormaps[main_dataset],alpha=alpha[main_dataset])
            else:
                axs[row_coronal,col].imshow(np.rot90(map_masked[main_dataset][:,max_y,:,i].T,2),vmin=lthresh, vmax=uthresh,cmap=colormaps[main_dataset],alpha=alpha[main_dataset])
            
            if len(self.k.keys())==2 and i<self.k[secondary_dataset]: # If maps present in both
                axs[row_coronal,col].imshow(np.rot90(map_masked[secondary_dataset][:,max_y,:,i].T,2),vmin=lthresh, vmax=uthresh,cmap=colormaps[secondary_dataset],alpha=alpha[secondary_dataset])
                if plot_overlap: # If we want to plot overlap
                    map1_bin = np.where(np.rot90(map_masked[main_dataset][:,max_y,:,i].T,2) > lthresh, 1, 0)
                    map2_bin = np.where(np.rot90(map_masked[secondary_dataset][:,max_y,:,i].T,2) > lthresh, 1, 0)
                    overlap = map1_bin * map2_bin      
                    contours = measure.find_contours(overlap)
                    for contour in contours:
                        axs[row_coronal,col].plot(contour[:,1],contour[:,0],linewidth=1,c='white',linestyle='solid',alpha=0.7)

            # Draw axial views
            row_axial = 1 if i<k_per_line else (i//k_per_line-1)*2+3
            axs[row_axial,col].axis('off');
            
            if len(self.k.keys())==2 and i<self.k[secondary_dataset]: # If maps present in both, define max based on the overlap
                overlap_map=self._overlap_maps(i,main_dataset,secondary_dataset)  
                max_z =int(np.where(overlap_map == np.nanmax(overlap_map))[2])
            else: # Otherwise, pick only based on the main dataset
                max_size=np.where(map_masked[main_dataset] == np.nanmax(map_masked[main_dataset][:,:,:,i]))[2].size
                if max_size>1:
                    max_z = int(np.where(map_masked[main_dataset] == np.nanmax(map_masked[main_dataset][:,:,:,i]))[2][int(max_size/2)]) # take the midle max if there are mainy
                else:
                    max_z = int(np.where(map_masked[main_dataset] == np.nanmax(map_masked[main_dataset][:,:,:,i]))[2])
            
            # Show template as background
            axs[row_axial,col].imshow(np.rot90(template_data[:,:,max_z].T,2),cmap='gray');
            
            # Show components
            axs[row_axial,col].imshow(np.rot90(map_masked[main_dataset][:,:,max_z,i].T,2),vmin=lthresh, vmax=uthresh,cmap=colormaps[main_dataset],alpha=alpha[main_dataset])
            
            if len(self.k.keys())==2 and i<self.k[secondary_dataset]: # If maps present in both                
                axs[row_axial,col].imshow(np.rot90(map_masked[secondary_dataset][:,:,max_z,i].T,2),vmin=lthresh, vmax=uthresh,cmap=colormaps[secondary_dataset],alpha=alpha[secondary_dataset])   
                
            # To "zoom" on the spinal cord, we adapt the x and y lims
            #axs[row_axial,col].set_xlim([map_masked[main_dataset].shape[0]*0.2,map_masked[main_dataset].shape[0]*0.8])
            #axs[row_axial,col].set_ylim([map_masked[main_dataset].shape[1]*0.2,map_masked[main_dataset].shape[1]*0.8])
            axs[row_axial,col].set_anchor('N')

        # If option is set, save results as a png
        if save_results == True:
            if len(self.k.keys())==2:
                plt.savefig(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_' + self.region + '_' + main_dataset + '_' + (self.subject[self.name1] if self.subject[self.name1] != None else 'group') + ('_' + self.duration[self.name1] if self.duration[self.name1] != None else '') + '_vs_' + secondary_dataset + '_' + (self.subject[self.name2] if self.subject[self.name2] != None else 'group') + ('_' + self.duration[self.name2] if self.duration[self.name2] != None else '') + '_thr' + str(lthresh)+ 'to' + str(uthresh) + '.png')
            elif len(self.k.keys())==1:
                print(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_' + self.region + '_' + main_dataset + '_' + (self.subject[self.name1] if self.subject[self.name1] != None else 'group') + ('_' + self.duration[self.name1] if self.duration[self.name1] != None else '') + '_thr' + str(lthresh)+ 'to' + str(uthresh) + '.png')
                plt.savefig(self.config['main_dir'] + self.config['output_dir'] + self.config['output_tag'] + '_' + self.region + '_' + main_dataset + '_' + (self.subject[self.name1] if self.subject[self.name1] != None else 'group') + ('_' + self.duration[self.name1] if self.duration[self.name1] != None else '') + '_thr' + str(lthresh)+ 'to' + str(uthresh) + '.png')
    
    
    # ======= BRAIN ========
        
    def br_plot(self, k_per_line=None, lthresh=2.3, uthresh=4.0, centering_method='max', colormap='autumn', direction='yxz', save_results=False):
        
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
        
    def _overlap_maps(self,k,main_dataset,secondary_dataset):
        mask1=np.where(~np.isnan(self.data_matched[main_dataset][:,:,:,k]),self.data_matched[main_dataset][:,:,:,k], np.nan)
        mask2=np.where(~np.isnan(self.data_sorted[secondary_dataset][:,:,:,k]),self.data_sorted[secondary_dataset][:,:,:,k], np.nan)
        overlap_map=(mask1+mask2)/2
        return overlap_map

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


class Plot_brain:
    def __init__(self, config):
        self.config = config # Load config info
        
    def plot_3D(self, i_img=None,hemi_view=["lh","rh"],face_view=["lateral"],vmin=None,vmax=None,threshold=1e-6, mask_img=None,colormap='hot', tag="",output_dir=None, save_results=False):
        '''
        This function help to plot functional 3D maps on a render surface 
        Two nilearn function are used, see details here:
        https://nilearn.github.io/dev/modules/generated/nilearn.surface.vol_to_surf.html
        https://nilearn.github.io/dev/modules/generated/nilearn.plotting.plot_surf_roi.html
        
        to do:
        - Add option to import directly a volume
        - Add new value possibilities for view_per_line
        
        Attributes
        ----------
        i_img <filename>: filename of the input 3D volume image (overlay) should be specify, default: False
        hemi_view <str>: hemisphere of the brain to display, one of the three options should be specified: ["lh","rh"] or ["lh"] or ["rh"]
        face_view <str> : must be a string in: "lateral","medial","dorsal", "ventral","anterior" or "posterior" multiple views can be specified simultaneously (e.g., ["lateral","medial"])
        vmin <float>, optional, Lower bound of the colormap. If None, the min of the image is used.
        vmax <float>, optional, upper bound of the colormap. If None, the min of the image is used. 
        threshold <int> or None optional. If None is given, the image is not thresholded. If a number is given, it is used to threshold the image: values below the threshold are plotted as transparent. If “auto” is given, the threshold is determined magically by analysis of the image. Default=1e-6.
        colormap <str> : specify a colormap for the plotting, default: 'automn'
        tag <str> : specify a tag name for the output plot filenames, default: '' 
        output_dir <directory name>, optional, set the output directory if None, the i_img directory will be used
        save_results <boolean>, optional, set True to save the results
        
        '''
        
        if i_img==None:
            raise Warning("Please provide the filename of the overlay volume (ex: i_img='/my_dir/my_func_img.nii.gz')")
        if output_dir==None:
            output_dir=os.path.dirname(i_img) + "/plots/"
        
        if save_results:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir) # create output folder
                os.mkdir(output_dir + "/tmp/") # create a temporary folder
     
        # 1. Select surface image for background --------------------------------------------------------
        surface_dir = self.config['templates']['surface_dir']
        
        img_surf={}
        for hemi in hemi_view:
            #2. Transform volume into surface image --------------------------------------------------------------------
            # include a mask is there are 0 values that you don't want to include in the interpolation
            img_surf[hemi]=surface.vol_to_surf(i_img,surface_dir+ hemi + ".pial",radius=0, 
                                     interpolation='nearest', kind='line', n_samples=10, mask_img=mask_img, depth=None)
  

            #3. Plot surface image --------------------------------------------------------------------
            side = "left" if hemi == "lh" else "right"
            for face in face_view:
                colorbar=True if face == face_view[-1] else False
                plot=plotting.plot_surf_roi(surface_dir+ hemi +".inflated", roi_map=img_surf[hemi],
                                            cmap=colormap, colorbar=colorbar,mask_img=mask_img,
                                            
                                            hemi=side, view=face,vmin=vmin,vmax=vmax,threshold=threshold,
                                            bg_map=surface_dir + hemi +".sulc",darkness=.5)

                if save_results:
                    
                    # Save each plot individually
                    plot.savefig(os.path.join(output_dir + "/tmp/", f'plot_{tag}_{hemi}_{face}.png'),dpi=150)
                    plt.close()

        if save_results:
            # Compute number of columns/rows and prepare subplots accordingly
            view_per_line= len(face_view)
               
            total_rows = ((len(face_view)*len(hemi_view))//view_per_line)
      
            fig, axs = plt.subplots(total_rows, view_per_line, figsize=(10*view_per_line, 8*total_rows))
                    
            for row, hemi in enumerate(hemi_view):
                for col, face in enumerate(face_view):
                    img = plt.imread(os.path.join(output_dir + "/tmp/", f'plot_{tag}_{hemi}_{face}.png'))
                    axs[row,col].imshow(img)
                    axs[row,col].axis('off')
        
            # Save the combined figure
            combined_save_path = os.path.join(output_dir, tag+".pdf")
            plt.savefig(combined_save_path, bbox_inches='tight')
            plt.show()
            
            #remove temporary files
            for hemi in hemi_view:
                for face in face_view:
                    os.remove(os.path.join(output_dir + "/tmp/", f'plot_{tag}_{hemi}_{face}.png'))
    
    
    
    