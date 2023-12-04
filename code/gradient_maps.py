# -*- coding: utf-8 -*-
import glob, os, json
from nilearn.maskers import NiftiMasker
from nilearn import image
import numpy as np
import statistics
import random
from collections import Counter

class GradientsMaps:
    '''
    The GradientsMaps class is used to create maps
    The maps are generate by computing for each brain voxels the spinal level gave the maximal MI value (from C1 to C7)
    
    Attributes
    ----------
    config : dict
    '''
    
    def __init__(self, config, mask=None):
        self.config = config # load config info
        self.seed_names=self.config["seeds"]["seed_names"] # name of your seeds or your condition
        self.IndivMapsDir= self.config["first_level"] # directory of the input data
        self.target=self.config["targeted_voxels"]["target_name"] # name of the target (i.e, where to extract the data could be brain ou spinalcord)
        self.subject_names= self.config["list_subjects"] # list of the participant to analyze
        self.mask_name=None # filename of the target mask (without directory either extension)
        self.secondlevel=self.config["second_level"] # directory to save the outputs
        
        if mask is not None:
            self.mask_file=self.config["main_dir"] + self.config["targeted_voxels"]["target_dir"] + mask + ".nii.gz"# mask to apply if needed for vizualisation purpose
            self.mask_name =  mask.split("_")[-1] # reduced name of the mask
            
            print(self.mask_name)
            
             
    def compute_GradMaps(self, methods="t_maps",output_tag="_",redo=False):
        '''
        methods: string
        could be either 
            - "MI": will use the MI for each individual and each levels
            
            - "t_maps": will use the t_maps from a statisitical analyse for each level
            second levels analyses sould be run first to extract t_values at the group level (for exemple using snpm)
        Attributes
        ----------
        config : dict
        '''
        self.output_tag=output_tag
        ##### 1. Ana info
        print("---------- Initialization info: ")
        for seed_nb in range(len(self.seed_names)):
            print(self.seed_names[seed_nb] + " will have a value of: " + str(seed_nb+1))

        #### 2. Create output diresctory:  -------------------------------------
        for seed_name in self.seed_names:
            self.output_dir=self.secondlevel +"/GradientsLevelsMaps/"
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)              

        #3 Select mask:
        self.mask_target=glob.glob(self.config["main_dir"] + self.config["targeted_voxels"]["target_dir"]+ self.target + ".nii.gz")[0] # mask of the voxels tareted for the an
        masker= NiftiMasker(self.mask_target,smoothing_fwhm=[0,0,0], t_r=1.55,low_pass=None, high_pass=None) # seed masker

        #### 4. Use MI individual maps  -------------------------------------
        
        if methods=="MI_indiv":
            tag="MI"
        elif methods=="Corr_indiv":
            tag="Corr"
        
        if methods == "MI_indiv" or methods=="Corr_indiv":
            MI_file={}; MI_data={}; file4D_data={}
            file4D=self.output_dir + "/4D_n" + str(len(self.subject_names)) + "_" + self.output_tag + "_"+tag+".nii.gz"

        
            if not os.path.exists(file4D) or redo==True:

                #4.a Select MI files and extract data
                for subject_name in self.subject_names:
                    MI_file[subject_name]=[]; MI_data[subject_name]=[];
                    
                    # select the map for a specific participant and for each seed (or condition)
                    for seed_nb in range(len(self.seed_names)):
                        MI_file[subject_name].append(glob.glob(self.IndivMapsDir + self.seed_names[seed_nb] + "/" + self.target + "_fc_maps/"+tag+"/pos-corr*" + subject_name +".nii.gz")[0]) # select individual maps
                        
                        MI_data[subject_name].append(masker.fit_transform(MI_file[subject_name][seed_nb])) # extract the data in a single array

                        
                #4.b Find max value in each voxels across the different individual maps
                for subject_name in self.subject_names:
                    data=np.array(MI_data[subject_name])
                    output_file=self.output_dir + "/sub-" + subject_name + "_" + output_tag +".nii.gz"
                    max_level_indices = []
                    for i in range(0,data.shape[2]):
                        i_values = data[:,:,i]  # Get the voxel values
                        if np.max(i_values)>=0: # here you can applye a threshold
                            max_level_index = np.argmax(i_values)  # Find the level that have the max value for this column (i.e across seeds or conditions)
                            max_level_indices.append(max_level_index+1) # add 1 to avoid 0 values (i.e attribute a value for each condition)
                        else: 
                            max_level_index = np.nan
                            max_level_indices.append(max_level_index) # add 1 to avoid 0 values

                    
                    #4.c Save the output as an image
                    masker= NiftiMasker(self.mask_target,smoothing_fwhm=[0,0,0]).fit()
                    seed_to_voxel_img = masker.inverse_transform(np.array(max_level_indices).T)
                    seed_to_voxel_img.to_filename(output_file) # create temporary 3D files
            
                #4.d transfome in 4D image and remove individual images
                file4D=self.output_dir + "/4D_n" + str(len(self.subject_names)) + "_" + output_tag + ".nii.gz"

                indiv_files=glob.glob(self.output_dir + "/sub-*") # concatenate the filename in a list
                new_list=",".join(indiv_files).replace(",", " ") # concatenate the filename in a list of files withou ',' as a delimiter
                string='fslmerge -t ' + file4D + " " + new_list # create an fsl command to merge the indiv files in a 4D file
                os.system(string) # run fsl command

                for file in indiv_files:
                    os.remove(file) # remove individual files

                #4.e Calculate the mean image
                mean_file=self.output_dir + "/mean_n" + str(len(self.subject_names)) + "_" + output_tag + ".nii.gz"
                if not os.path.exists(mean_file) or redo==True:
                    string="fslmaths " + file4D + " -Tmean " + mean_file # create an fsl command to calculate eman value 
                    os.system(string) # run fsl command

                    #mask the image
                    if self.mask_name is not None:
                        mean_maskfile=self.output_dir + "/mean_n" + str(len(self.subject_names)) + "_" + output_tag +"_"+self.mask_name+".nii.gz"
                        string='fslmaths ' + mean_file + ' -mas ' + self.mask_file + " " + mean_maskfile # fsl command to mask
                        os.system(string) # run fsl command
            
            #4.f Calulate the median
                median_file=self.output_dir + "/median_n" + str(len(self.subject_names)) + "_" + output_tag + ".nii.gz"
                if not os.path.exists(median_file) or redo==True:
                    string="fslmaths " + file4D + " -Tmedian  " + median_file # create an fsl command to calculate eman value 
                    os.system(string) # run fsl command

                    #mask the image
                    if self.mask_name is not None:
                        median_maskfile=self.output_dir + "/median_n" + str(len(self.subject_names)) + "_" + output_tag +"_"+self.mask_name+".nii.gz"
                        string='fslmaths ' + median_file + ' -mas ' + self.mask_file + " " + median_maskfile # fsl command to mask
                        os.system(string) # run fsl command

            
            #4.g Calulate the mode
            file4D_data=np.array(masker.fit_transform(file4D)) # extract the data in a single array
            
            mode_values=[]
            for i in range(0,file4D_data.shape[1]):
                
                # Count the occurrences of each value
                value_counts = Counter(file4D_data[:,i])
                max_frequency = max(value_counts.values())# Find the maximum frequency
                modes = [value for value, frequency in value_counts.items() if frequency == max_frequency] # Find all the values with the maximum frequency
                
                if max_frequency>=1: # you can choose a threshold
                    if len(modes)>1:
                        mode_value = np.mean(modes)#random.choice(modes) # Select one mode randomly

                    else:
                        mode_value=modes[0]
                else:
                        mode_value=-1
                mode_values.append(mode_value)
                
                
            #4.c Save the output as an image
            output_file=self.output_dir + "/mode_n" + str(len(self.subject_names)) + "_" + output_tag + ".nii.gz"
            masker= NiftiMasker(self.mask_target,smoothing_fwhm=[0,0,0]).fit()
            img = masker.inverse_transform(np.array(mode_values).T)
            img.to_filename(output_file) # create temporary 3D files
            
             #mask the image
            if self.mask_name is not None:
                mode_maskfile=self.output_dir + "/mode_n" + str(len(self.subject_names)) + "_" + output_tag +"_"+self.mask_name+".nii.gz"
                string='fslmaths ' + output_file+ ' -mas ' + self.mask_file + " " + mode_maskfile # fsl command to mask
                os.system(string) # run fsl command

            
                
        elif methods=="MI_group":
            MIgroup_file=[];MIgroup_data=[]
            for seed_nb in range(len(self.seed_names)):
                MIgroup_file.append(glob.glob(self.IndivMapsDir + self.seed_names[seed_nb] + "/" + self.target + "_fc_maps/MI/mi*mean.nii.gz")[0]) # select individual maps
                MIgroup_data.append(masker.fit_transform( MIgroup_file[seed_nb])) # extract the data in a single array

                data=np.array(MIgroup_data)
                output_file=self.output_dir + "/mean2_" + output_tag +".nii.gz"
                max_level_indices = []
                for i in range(0,data.shape[2]):
                    i_values = data[:,:,i]  # Get the voxel values
                    max_level_index = np.argmax(i_values)  # Find the level that have the max value for this column
                    max_level_indices.append(max_level_index+1) # add 1 to avoid 0 values

                ####5. Output Image
                #5.a Save the output as an image
                masker= NiftiMasker(self.mask_target,smoothing_fwhm=[0,0,0]).fit()
                seed_to_voxel_img = masker.inverse_transform(np.array(max_level_indices).T)
                seed_to_voxel_img.to_filename(output_file) # create temporary 3D files

                
            
        #####5 Find the t max value for each voxels (group level) ________________________
        elif methods=="t_maps":
            tmaps_file=[];tmaps_data=[]
            for seed_nb in range(len(self.seed_names)):
                #print(self.secondlevel + "/GLM/OneSampleT/" + self.target + "/Corr/" +self.seed_names[seed_nb]+ "/IP_FWE+.img")
                tmaps_file.append(glob.glob(self.secondlevel + "/GLM/OneSampleT/" + self.target + "/Corr/" +self.seed_names[seed_nb]+ "/uncorr/zscore_*")[0]) # select individual maps   
                tmaps_thr=image.threshold_img(tmaps_file[seed_nb], threshold=1.3, cluster_threshold=200, mask_img=self.mask_target)
                tmaps_thr.to_filename(self.secondlevel+ "/GLM/OneSampleT/" +self.seed_names[seed_nb] +"_thr_t1.nii.gz") # create temporary 3D files
                tmaps_data.append(masker.fit_transform(tmaps_thr)) # extract the data in a single array

                data=np.array(tmaps_data)
                output_file=self.output_dir + "/tmax_" + output_tag +".nii.gz"
                max_level_indices = []
                
                for i in range(0,data.shape[2]):
                    i_values = data[:,:,i]  # Get the voxel values

                    max_level_index = np.argmax(i_values )  # Find the level that have the max value for this column
                    if i_values[max_level_index] == 0 :
                        max_level_index =-1 # if the max value is 0 put -1 to the index
                           
                    max_level_indices.append(max_level_index+1) # add 1 to avoid 0 values

                ####5. Output Image
                #5.a Save the output as an image
                masker= NiftiMasker(self.mask_target,smoothing_fwhm=[0,0,0]).fit()
                seed_to_voxel_img = masker.inverse_transform(np.array(max_level_indices).T)
                seed_to_voxel_img.to_filename(output_file) # create temporary 3D files

        #6. copy the config file
        with open(self.secondlevel + '/' + output_tag + '_analysis_config.json', 'w') as fp:
            json.dump(self.config, fp)

   
    
    