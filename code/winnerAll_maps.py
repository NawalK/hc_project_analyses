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
    
    def __init__(self, config, mask=None):
        self.config = config # load config info
        self.seed_names=self.config["gradient_seeds"] # name of your seeds or your condition
        #self.IndivMapsDir= self.config["first_level"] # directory of the input data
        self.subject_names= self.config["list_subjects"] # list of the participant to analyze
        self.mask=self.config["winner_all"]["mask"]# filename of the target mask (without directory either extension)
        self.secondlevel=self.config["second_level"] # directory to save the outputs
        self.indir=self.secondlevel+ self.config["winner_all"]["input_dir"] # directory to save the outputs
        self.tag_input=self.config["winner_all"]["tag_input"]
        self.analysis=self.config["winner_all"]["analysis"]
        
        self.mask_name =  self.mask.split("/")[-1].split(".")[0] # reduced name of the mask
            
        print(self.mask_name)
            
             
    def compute_GradMaps(self, output_tag="_",apply_threshold=None,redo=False):
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

   
    
    