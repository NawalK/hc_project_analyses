# -*- coding: utf-8 -*-
import glob, os, json
from nilearn.maskers import NiftiMasker
from nilearn import image
import numpy as np
import statistics
import random
from collections import Counter

class WinnerAll:
    '''
    The WinnerAll class is used to compute winner take all analaysis
    The maps are generate by computing for each brain voxels the spinal level gave the maximal value (from C1 to C7)
    
    Attributes
    ----------
    config : dict
    '''
    
    def __init__(self, config, mask=None,verbose=True):
        self.config = config # load config info
        self.seed_names=self.config["gradient_seeds"] # name of your seeds or your condition
        #self.IndivMapsDir= self.config["first_level"] # directory of the input data
        self.subject_names= self.config["list_subjects"] # list of the participant to analyze
        self.mask=self.config["winner_all"]["mask"]# filename of the target mask (without directory either extension)
        self.secondlevel=self.config["second_level"] # directory to save the outputs
        self.indir=self.config["winner_all"]["input_dir"] # directory to save the outputs
        self.tag_input=self.config["winner_all"]["tag_input"]
        self.analysis=self.config["winner_all"]["analysis"]
        
        self.mask_name =  self.mask.split("/")[-1].split(".")[0] # reduced name of the mask
        if verbose==True:    
            print("Analyses will be run in the following mask:" + self.mask_name)
            
             
    def compute_GradMaps(self, output_tag="_",apply_threshold=None,redo=False,verbose=True):
        '''
        Use se the t_maps or mean group 
        Attributes
        ----------
        config : dict
        
        apply_threshold: str (default None)
        To apply a threshold value at the input images, a cluster thresholding of 100 will also be applied
        
        '''
        self.output_tag=output_tag
        ##### 1. Ana info
        if verbose==True:
            print("---------- Initialization info: ")
            for seed_nb in range(len(self.seed_names)):
                print(self.seed_names[seed_nb] + " will have a value of: " + str(seed_nb+1))

        #### 2. Create output diresctory:  -------------------------------------
        for seed_name in self.seed_names:
            self.output_dir=self.secondlevel +"/WinnerTakeAll/" + self.analysis
            
            if not os.path.exists(self.secondlevel +"/WinnerTakeAll/"):
                os.mkdir(self.secondlevel +"/WinnerTakeAll/") # create main directory
                
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir) # create sub directory for each analysis

        #3 Select mask:
        masker= NiftiMasker(self.mask,smoothing_fwhm=[0,0,0], t_r=1.55,low_pass=None, high_pass=None) # seed masker

        #4. Find the t max value for each voxels (group level) ________________________
        maps_file=[];maps_data=[]
        for seed_nb in range(len(self.seed_names)):
            #print(self.indir +self.seed_names[seed_nb]+ self.tag_input)
            maps_file.append(glob.glob(self.indir +self.seed_names[seed_nb]+ self.tag_input)[0]) # select individual maps   

            if apply_threshold is not None:
                maps_thr=image.threshold_img(maps_file[seed_nb], threshold=apply_threshold, cluster_threshold=100, mask_img=self.mask)

                maps_thr.to_filename(maps_file[seed_nb].split(".")[0] +"_thr_t"+str(apply_threshold)+".nii.gz") # create temporary 3D files
                maps_data.append(masker.fit_transform(maps_thr)) # extract the data in a single array

            elif apply_threshold is None:
                maps_data.append(masker.fit_transform(maps_file[seed_nb])) # extract the data in a single array

            data=np.array(maps_data)
            output_file=self.output_dir + "/" + output_tag +".nii.gz"
            max_level_indices = []
                
            for i in range(0,data.shape[2]):
                i_values = data[:,:,i]  # Get the voxel values

                max_level_index = np.argmax(i_values )  # Find the level that have the max value for this column
                if i_values[max_level_index] == 0 :
                    max_level_index =-1 # if the max value is 0 put -1 to the index

                max_level_indices.append(max_level_index+1) # add 1 to avoid 0 values

            ####5. Output Image
            #5.a Save the output as an image
            seed_to_voxel_img = masker.inverse_transform(np.array(max_level_indices).T)
            seed_to_voxel_img.to_filename(output_file) # create temporary 3D files

        #6. copy the config file
        with open(self.output_dir + '/' + output_tag + '_analysis_config.json', 'w') as fp:
            json.dump(self.config, fp)

   
    
    def group_indiv_GradMaps(self,output_tag,redo=False):
    
        #4.d transfome in 4D image and remove individual images
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
    