# -*- coding: utf-8 -*-
import glob, os
import sc_utilities as util
import matlab.engine
import pandas as pd
import numpy as np

# nilearn:
from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_design_matrix
from nilearn import plotting
from nilearn.image import get_data, math_img
from nilearn.glm import threshold_stats_img

import matplotlib.pyplot as plt

# stats
from scipy.stats import norm

class Stats:
    '''
    The Stats class is used to run group level analysis
    Attributes
    ----------
    config : dict
    measure: str
        first level measure could be "MI" or "Corr"
    '''
    
    def __init__(self, config,measure):
        self.config = config # load config info
        self.measure=measure
        self.subject_names= config["list_subjects"]
        self.outputdir= self.config["main_dir"] +self.config["seed2vox_dir"]
        self.seed_names=self.config["seeds"]["seed_names"]
        self.target=self.config["targeted_voxels"]["target_name"]
        self.seed_structure=self.config["seeds"]["seed_structure"]
        self.target_structure=self.config["targeted_voxels"]["target_structure"]
        
        #>>> create output directory if needed -------------------------------------
        for seed_name in self.seed_names:
            os.makedirs(self.config['main_dir'] + self.config['seed2vox_dir'] + '/2_second_level/'+seed_name+'/'+ self.target +'/' + self.measure,exist_ok=True) # folder to store maps of FC
        
        

    
        #>>> Select mask: -------------------------------------
        self.mask_target=glob.glob(self.config["main_dir"] + self.config["targeted_voxels"]["target_dir"]+ self.target + ".nii.*")[0] # mask of the voxels tareted for the analysis
        if self.mask_target.split(".")[-1] =="gz":
            util.unzip_file(self,[self.mask_target],ext=".nii",zip_file=False, redo=False) # unzip files
            self.mask_target=self.mask_target.split(".")[0] + ".nii"
        
        print("Start the analysis on: " + str(len(self.subject_names))+ " participants")
        print("targeted voxel's group mask: " + self.target)
        #print(self.mask_target)
        
        
        #>>> Select first level data: -------------------------------------
        self.data_1rstlevel=[];
        for sbj_nb in range(len(config['list_subjects'])):
            subject_name='sub-' +  config['list_subjects'][sbj_nb]
            tag_files = {"Corr": "corr","MI": "mi"}
            self.tag_file = tag_files.get(measure, None)
            
            
            self.data_1rstlevel.append(glob.glob(self.config["first_level"] +'/'+ seed_name+'/'+ self.target +'_fc_maps/'+ measure + "/" + self.tag_file + "*" + subject_name + "_z.nii.gz")[0])
        print(self.data_1rstlevel)
        
        #>>>> unzip the data if they are zipped (for SPM)
        #if self.data_1rstlevel[sbj_nb].split(".")[-1] =="gz":
         #   util.unzip_file(self,self.data_1rstlevel,ext=".nii",zip_file=False, redo=False) # unzip files
          #  for sbj_nb in range(len(config['list_subjects'])):
           #     os.remove(self.data_1rstlevel[sbj_nb]) # remove the zipped files
            #    self.data_1rstlevel[sbj_nb]=self.data_1rstlevel[sbj_nb].split(".")[0]
                
    def OneSampT(self,seed_name,z_thr=1,p_value=0.05,corr=None,smoothing=None,parametric=True,plot_matrix=False,plot_stats_uncorr=False):
        
        # Create output directory:
        if parametric == True:
            output_dir=self.config['main_dir'] + self.config['seed2vox_dir'] + '/2_second_level/'+seed_name+'/'+ self.target +'/' +self.measure +'/OneSampT_param/'
        else:
            output_dir=self.config['main_dir'] + self.config['seed2vox_dir'] + '/2_second_level/'+seed_name+'/'+ self.target +'/' +self.measure +'/OneSampT_noparam/'  
        os.makedirs(output_dir,exist_ok=True)
        
        input_dir=self.config["first_level"] +'/'+ seed_name+'/'+ self.target +'_fc_maps/'+ self.measure + "/"
        
        # >>>>> here try to use the 4D instead of the 3D
        second_level_input=glob.glob(input_dir + self.tag_file + "_sub-*_z.nii.gz")
        
        # Create a design matrix. For one sample T test we need a single column of ones, corresponding to the model intercept
        design_matrix = pd.DataFrame([1] * len(second_level_input),columns=["intercept"])
        if plot_matrix== True:
            ax = plot_design_matrix(design_matrix)
            ax.set_title("Second level design matrix", fontsize=12)
            ax.set_ylabel("maps")
            plt.tight_layout()
            plt.show()

        # Specify the model
        second_level_model = SecondLevelModel(smoothing_fwhm=smoothing)
        second_level_model = second_level_model.fit(second_level_input,design_matrix=design_matrix,)
        
        # estimate the constrast
        z_map = second_level_model.compute_contrast(second_level_contrast="intercept",output_type="z_score",)
        
        # Statistics
        thresholded_map1, threshold1 = threshold_stats_img(z_map,alpha=p_value,threshold=z_thr,height_control=corr,cluster_threshold=10,two_sided=False)

        if plot_stats_uncorr==True:
            self._plot_stats(thresholded_map1,seed_name + " " + self.measure ,threshold1)
            
        # Correct stats
        #p_val = second_level_model.compute_contrast(output_type="p_value")
        #n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
        # Correcting the p-values for multiple testing and taking negative logarithm
        #neg_log_pval = math_img(f"-np.log10(np.minimum(1, img * {str(n_voxels)}))",img=p_val,)
        
    def Snpm_OneSampT(self,seed_name,permutation=100,t_thr=3.1):

        os.chdir('../../code/stats/')# need to change the directory to find the SPM function
        output_dir=self.config['main_dir'] + self.config['seed2vox_dir'] + '/2_second_level/'+seed_name+'/'+ self.target +'/' +self.measure +'/SnPM_OneSampT/'
        os.makedirs(output_dir,exist_ok=True)
        
        input_dir=self.config["first_level"] +'/'+ seed_name+'/'+ self.target +'_fc_maps/'+ self.measure + "/"
        
        
        eng = matlab.engine.start_matlab()
        print("Number of permutation: " + str(permutation))
        print("Threshold: " + str(t_thr))
        print(eng.SnPM_OneSampT_job(permutation,t_thr,input_dir, self.tag_file ,self.mask_target,output_dir,seed_name+ "_FWE")) #for quality check
        
        
    def _plot_stats(self,second_level_map,title,threshold):
        #p001_unc = norm.isf(p_val)
        display = plotting.plot_glass_brain(
            second_level_map,
            colorbar=True,
            threshold=threshold,
            display_mode='lyrz',
            title=title,)
        plotting.show()
        