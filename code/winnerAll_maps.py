# -*- coding: utf-8 -*-
import glob, os, json
from nilearn.maskers import NiftiMasker
from nilearn import image
import numpy as np
import statistics
import random
from collections import Counter
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WinnerAll:
    '''
    The WinnerAll class is used to compute winner take all analaysis
    The maps are generate by computing for each brain voxels the spinal level gave the maximal value (from C1 to C7)
    
    Attributes
    ----------
    config : dict
    '''
    
    def __init__(self, config, indiv=False,mask=None,verbose=True):
        self.config = config # load config info
        self.indiv=indiv
        self.seed_names=self.config["gradient_seeds"] # name of your seeds or your condition
        #self.IndivMapsDir= self.config["first_level"] # directory of the input data
        self.subject_names= self.config["list_subjects"] # list of the participant to analyze
        
        self.wta_dir=self.config["main_dir"] + self.config["wta_dir"] # directory to save the outputs
        if indiv==False:
            self.indir=self.config["main_dir"] + self.config["winner_all"]["input_dir"] # directory to save the inputs
            self.tag_input=self.config["winner_all"]["tag_input"]
            self.analysis=self.config["winner_all"]["analysis"]
            self.mask=self.config["winner_all"]["mask"]# filename of the target mask (without directory either extension)
        else:
            self.indir==self.config["main_dir"] + self.config["winner_all_indiv"]["input_dir"]
            self.tag_input=self.config["winner_all_indiv"]["tag_input"]
            self.analysis=self.config["winner_all_indiv"]["analysis"]
            self.mask=self.config["winner_all_indiv"]["mask"]# filename of the target mask (without directory either extension)
            
        self.mask_name =  self.mask.split("/")[-1].split(".")[0] # reduced name of the mask
        if verbose==True:    
            print("Analyses will be run in the following mask:" + self.mask_name)
            
             
    def compute_GradMaps(self, output_tag="_",fwhm=[0,0,0],cluster_threshold=100,apply_threshold=None,redo=False,verbose=True):
        '''
        Use se the t_maps or mean group 
        Attributes
        ----------
        config : dict
        fwhm <scalar>: smoothing will be apply in 3 dimensions with gaussian filter
        apply_threshold: str (default None)To apply a threshold value at the input images, 
        cluster_threshold <str>: to appl a cluster thresholding (default: 100)
        
        '''
        self.output_tag=output_tag
        ##### 1. Ana info
        if verbose==True:
            print("---------- Initialization info: ")
            for seed_nb in range(len(self.seed_names)):
                print(self.seed_names[seed_nb] + " will have a value of: " + str(seed_nb+1))
        
        
        #### 2. Create output diresctory:  -------------------------------------
        for seed_name in self.seed_names:
            self.output_dir=self.wta_dir +"/WinnerTakeAll/" + self.analysis
            
            if not os.path.exists(self.wta_dir +"/WinnerTakeAll/"):
                os.mkdir(self.wta_dir +"/WinnerTakeAll/") # create main directory
                
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir) # create sub directory for each analysis
        
        

        #3 Select mask for individual maps (no smoothing):
        
        masker= NiftiMasker(self.mask,smoothing_fwhm=[0,0,0], t_r=1.55,low_pass=None, high_pass=None) # seed masker
            
        if self.indiv:
            max_level_indices=[];output_indiv_file=[];output_files=[]
            output_4d_file=self.output_dir +  "/4d_"+ output_tag + "_thr" + str(apply_threshold)+"_cluster"+ str(cluster_threshold) + "_s"+str(fwhm[0]) + ".nii.gz"
            for seed_nb, seed_name in enumerate(self.seed_names):
                output_file=self.output_dir +  "/"+ seed_name + "_"+ output_tag + "_thr" + str(apply_threshold)+"_cluster"+ str(cluster_threshold) + "_s"+str(fwhm[0]) + "_distr.nii.gz"
                    
                output_files.append(output_file)

            if not os.path.exists(output_4d_file):
                for ID_nb, ID in enumerate(self.subject_names):
                    output_indiv_file.append(self.output_dir + "/sub-"+ID+"_"+ output_tag + "_thr" + str(apply_threshold)+"_cluster"+ str(cluster_threshold) + "_s"+ str(fwhm[0]) + ".nii.gz")
                    # Calculate the level for each voxel in each participant
                    max_level_indices.append(self._WTA_array(output_file=output_indiv_file[ID_nb],
                                                             masker=masker,apply_threshold=apply_threshold,cluster_threshold=cluster_threshold,fwhm=fwhm,ID=ID))
                    
                    
            # Merge indiv files and remove indiv files
                list_f=" ".join(output_indiv_file)
                string="fslmerge -t " + output_4d_file + " " + list_f
                os.system(string)
                for file in output_indiv_file:
                    os.remove(file)
                masker.fit_transform(output_4d_file)
            
            # Calculate the distribution for each K
                k_maps = np.zeros((7,170630))
                for seed_nb, seed_name in enumerate(self.seed_names):
                    output_file=self.output_dir +  "/"+ seed_name + "_"+ output_tag + "_thr" + str(apply_threshold)+"_cluster"+ str(cluster_threshold) + "_s"+str(fwhm[0]) + "_distr.nii.gz"
                    
                    for ID_nb, ID in enumerate(self.subject_names):
                        for vox in range(0,170630):
                            k_maps[seed_nb, vox] += (max_level_indices[ID_nb][vox] == seed_nb+1)
                    labels_img = masker.inverse_transform(k_maps[seed_nb,:])
                    labels_img.to_filename(output_file)
            
            
           
        else:
            output_file=self.output_dir + "/" + output_tag + "_thr" + str(apply_threshold)+"_cluster"+ str(cluster_threshold) + "_s"+ str(fwhm[0]) + ".nii.gz"
            if not os.path.exists(output_file) or redo==True:
                max_level_indices=self._WTA_array(output_file=output_file,masker=masker,apply_threshold=apply_threshold,cluster_threshold=cluster_threshold,fwhm=fwhm,ID=None)
                #6. copy the config file
                with open(self.output_dir + '/' + output_tag + '_analysis_config.json', 'w') as fp:
                    json.dump(self.config, fp)

        return output_files if self.indiv else output_file
    
    
    def voxel_distr_GradMaps(self,input_file=None,plot=True,save_plot=False,redo=False):
        # Create directory
        if not os.path.exists(self.output_dir + '/vox_distr/'):
            os.mkdir(self.output_dir + '/vox_distr/')
                              
        
        # 1. import the fMRI data:
        img_4d = nib.load(input_file)
        data = img_4d.get_fdata() # extract data
        fmri_flat = data.reshape(-1, data.shape[-1]) # Flatten the fMRI data
        
        # 2. Load the mask data for each seed
        df = pd.DataFrame(columns=['IDs','level_assigned','Mask','Total_vox','Percentage'])

        for seed_nb, seed_name in enumerate(self.seed_names):
            mask_file=glob.glob(self.output_dir+"/masks/*" + seed_name + "*")[0] # select seed mask image
            mask_img = nib.load(mask_file) # load the mask image
            mask_data = mask_img.get_fdata() # extract data
            
            # 3. extract the fmri value in the mask
            mask_flat = mask_data.flatten() # Flatten the mask to make indexing easier
            masked_data = fmri_flat[mask_flat.astype(bool)] # Extract values from fMRI data within the mask
            # Specity the value of interst (i.e the seeds)
            values_of_interest = [1, 2, 3, 4, 5, 6, 7]
            
            num_voxels_mask = np.sum(mask_flat.astype(bool)) # total number of voxels in the mask

            participant_names = []; value_names = []; percentages = [] ;  total_voxels=[]; seed_names=[] # initiate variables
            print(seed_name )
            print(num_voxels_mask)
            print(" ")
            # Calculate percentage of voxels for each value of interest
            for value in values_of_interest:
                
                num_voxels_interest = np.sum(masked_data == value, axis=0) # calculate the number of voxels having the required value
                percentage = (num_voxels_interest / num_voxels_mask) * 100 # calculate the percentage of voxels having the required value
                for i, (perc, num_voxels) in enumerate(zip(percentage, num_voxels_interest)):
                    participant_names.append("sub-" + self.config["list_subjects"][i])
                    value_names.append(f'C{value}')
                    percentages.append(perc)
                    total_voxels.append(num_voxels)
                    seed_names.append(seed_name)
        
                    
            # Create a DataFrame to store the percentages for a specific mask
            df_mask = pd.DataFrame({'IDs': participant_names,
                                    'level_assigned': value_names,
                                    'Total_vox': total_voxels,
                                    'Percentage': percentages,
                                    'Mask': seed_name })
            
            df= pd.concat([df, df_mask], ignore_index=True)
            
        # Save the DataFrame to a CSV file
        output_csv_file = self.output_dir + '/vox_distr/percentages_all_seeds.csv'
        df.to_csv(output_csv_file)
        
        if plot:
            # Iterate over each value of interest
            
            for seed_nb, seed_name in enumerate(self.seed_names):
                palette=["#1a04a4",'#0070ff','#07f6e0', "#9bff00",'#e8f703', '#fa9c03', '#ff3a00']
                df_filtered = df[df['Mask'] == seed_name]

                SEM = df_filtered.groupby('level_assigned')['Total_vox'].sem() # Calculate the standard deviation of the mean for each mask
                
                # Plot the results using seaborn
                plt.figure(figsize=(4, 5))
                ax=sns.pointplot(data=df_filtered, x='level_assigned', y='Total_vox', palette=palette, hue='level_assigned', estimator=np.mean,errorbar=None)
                plt.errorbar(x=SEM.index, y=df_filtered.groupby('level_assigned')['Total_vox'].mean(), yerr=SEM, fmt='o', color="#babbc6", capsize=2, markerfacecolor='green', markersize=1, zorder=0)
                
                # Remove spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set(ylim=(0,600))
                plt.xlabel('Spinal level assigned')
                plt.ylabel('Total number of voxels')
                plt.title(f'Total number of voxels of voxels in {seed_name}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                
                # Save the figure as a PDF
                if save_plot:
                    output_filename = self.output_dir + "/vox_distr/" + f'plot_C{str(seed_nb+1)}_gm_rescale.svg'

                    plt.savefig(output_filename, format='svg')
                plt.show()
                plt.close()  # Close the figure to release memory



        return df
    
    def group_indiv_GradMaps(self,output_tag,redo=False):
    
    
        #4.d transfome in 4D image and remove individual images
        self.output_dir=self.wta_dir +"/WinnerTakeAll/" + self.analysis
        file4D=self.output_dir + "/4D_n" + str(len(self.subject_names)) + "_" + output_tag + ".nii.gz"
        indiv_files=glob.glob(self.output_dir + "/sub-*.nii.gz") # concatenate the filename in a list
        indiv_json_files=glob.glob(self.output_dir + "/sub-*.json") # concatenate the filename in a list
    
        new_list=",".join(indiv_files).replace(",", " "); # concatenate the filename in a list of files withou ',' as a delimiter
        string='fslmerge -t ' + file4D + " " + new_list # create an fsl command to merge the indiv files in a 4D file
        os.system(string) # run fsl command

        for file in indiv_json_files:
            os.remove(file) # remove individual files
        
        for file in indiv_files:
            os.remove(file) # remove individual files

        #4.e Calculate the mean image
        mean_file=self.output_dir + "/mean_n" + str(len(self.subject_names)) + "_" + output_tag + ".nii.gz"
        if not os.path.exists(mean_file) or redo==True:
            string="fslmaths " + file4D + " -Tmean " + mean_file # create an fsl command to calculate eman value 
            os.system(string) # run fsl command

            #mask the image
            if self.mask is not None:
                masker= NiftiMasker(self.mask,smoothing_fwhm=[0,0,0], t_r=1.55,low_pass=None, high_pass=None) # seed masker
                mean_maskfile=self.output_dir + "/mean_n" + str(len(self.subject_names)) + "_" + output_tag +"_"+self.mask+".nii.gz"
                string='fslmaths ' + mean_file + ' -mas ' + self.mask + " " + self.mask # fsl command to mask
                os.system(string) # run fsl command
            
            #4.f Calulate the median
            median_file=self.output_dir + "/median_n" + str(len(self.subject_names)) + "_" + output_tag + ".nii.gz"
            if not os.path.exists(median_file) or redo==True:
                string="fslmaths " + file4D + " -Tmedian  " + median_file # create an fsl command to calculate eman value 
                os.system(string) # run fsl command

            #mask the image
            if self.mask is not None:
                median_maskfile=self.output_dir + "/median_n" + str(len(self.subject_names)) + "_" + output_tag +"_"+self.mask+".nii.gz"
                string='fslmaths ' + median_file + ' -mas ' + self.mask + " " + median_maskfile # fsl command to mask
                os.system(string) # run fsl command

            
            #4.g Calulate the mode
            file4D_data=np.array(masker.fit_transform(file4D)) # extract the data in a single array
            
            mode_values=[]
            for i in range(0,file4D_data.shape[1]):
                
                # Count the occurrences of each value
                value_counts = Counter(file4D_data[:,i])
                max_frequency = max(value_counts.values())# Find the maximum frequency
                modes = [value for value, frequency in value_counts.items() if frequency == max_frequency] # Find all the values with the maximum frequency
                
                if max_frequency>=5: # you can choose a threshold
                    if len(modes)>1:
                        mode_value = np.mean(modes)#random.choice(modes) # Select one mode randomly

                    else:
                        mode_value=modes[0]
                else:
                        mode_value=-1
                mode_values.append(mode_value)
                
                
                
            #4.c Save the output as an image
            output_file=self.output_dir + "/mode_n" + str(len(self.subject_names)) + "_" + output_tag + ".nii.gz"
            print(output_file)
            img = masker.inverse_transform(np.array(mode_values).T)
            img.to_filename(output_file) # create temporary 3D files
            
            #mask the image
            if self.mask_name is not None:
                mode_maskfile=self.output_dir + "/mode_n" + str(len(self.subject_names)) + "_" + output_tag +"_"+self.mask+".nii.gz"
                string='fslmaths ' + output_file+ ' -mas ' + self.mask + " " + mode_maskfile # fsl command to mask
                os.system(string) # run fsl command
    
       
        
    def _WTA_array(self,output_file=None,masker=None,apply_threshold=None,cluster_threshold=None,fwhm=[0,0,0],ID=None):
        
        maps_file=[];maps_data=[]
        for seed_nb, seed_name in enumerate(self.seed_names):
            if ID==None:
                maps_file.append(glob.glob(self.indir +seed_name+ self.tag_input)[0]) # select individual maps
            else:
                maps_file.append(glob.glob(self.indir +self.seed_names[seed_nb]+ self.tag_input + "sub-" + ID +"*")[0]) # concatenate the filename in a list
                                 
            # Apply threshold to the inputs images if necessary
            if apply_threshold == None:
                maps_data.append(masker.fit_transform(maps_file[seed_nb])) # extract the data in a single array
            else:
                maps_thr=image.threshold_img(maps_file[seed_nb], threshold=apply_threshold, cluster_threshold=cluster_threshold, mask_img=self.mask)
                maps_thr.to_filename(maps_file[seed_nb].split(".")[0] +"_thr_t"+str(apply_threshold)+".nii.gz") # create temporary 3D files
                maps_data.append(masker.fit_transform(maps_thr)) # extract the data in a single array
              
        # Create an array with all seed maps
        data=np.array(maps_data)
        max_level_indices = []
        
        for i in range(0,data.shape[2]):
            i_values = data[:,:,i]  # Get the voxel values

            max_level_index = np.argmax(i_values)  # Find the level that have the max value for this column
            if i_values[max_level_index] == 0 :
                max_level_index =np.nan # if the max value is 0 put -1 to the index

            max_level_indices.append(max_level_index+1) # add 1 to avoid 0 values
                                 
        ####Output Image
        #5.a Save the output as an image, group level, smoothing can be applied if needed
        seed_to_voxel_img = masker.inverse_transform(np.array(max_level_indices).T)
        if fwhm!=[0,0,0]:
            seed_to_voxel_img_s=image.smooth_img(seed_to_voxel_img,fwhm)
            seed_to_voxel_img=seed_to_voxel_img_s

            seed_to_voxel_img.to_filename(output_file) # create temporary 3D files
            # threshold the output image to avoid unrelevant values
            string="fslmaths "+ output_file + " -thr 0.8 " + output_file
            os.system(string)

        else:
            seed_to_voxel_img.to_filename(output_file) # create temporary 3D files
         
        
        return max_level_indices