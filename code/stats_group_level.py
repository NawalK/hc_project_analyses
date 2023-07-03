# -*- coding: utf-8 -*-
import glob, os, json, math
import sc_utilities as util
import matlab.engine
import pandas as pd
import numpy as np
import nibabel as nb
# nilearn:
from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_design_matrix
from nilearn import plotting, image
from nilearn.image import get_data, math_img
from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table
from nilearn.glm.second_level import non_parametric_inference

import matplotlib.pyplot as plt

# stats
from scipy.stats import norm

class Stats:
    
    
    def __init__(self, config,ana_name,save_ana=False):
        '''
        The Stats class will initiate the group level analysis
        I. Initiate variable
        II. Create output directory (if save_ana=True)
        III. Select first level data
        
        Attributes
            ----------
        config : dict
        measure: str
        first level measure could be "MI" or "Corr"
        '''
        
        #>>> I. Initiate variable -------------------------------------
        self.config = config # load config info
        self.ana_name=ana_name
        self.measure=self.config["measure"]
        self.model=self.config["model"] ##OneSampleT #TwoSampT_paired or #TwoSampT_unpaired or "HigherOrder_paired"
        self.mask_img=self.config["mask_path"]# if no mask was provided the whole target image will be used
        self.seed_names=self.config["seeds"][ana_name] # seeds to include in the analysis
        self.subject_names= self.config["list_subjects"]
        self.outputdir= self.config["main_dir"] +self.config["seed2vox_dir"]
        self.target=self.config["target_1rstlevel"]
        
        self.output_tag=self.ana_name
        print("************************************** ")
        print("Initiate " + self.ana_name + " analysis")
        print("  ")
        print("> Statistical model: " +self.model)
        print("> Number of participants: "+ str(len(self.subject_names)))
        print("> Mask : " + os.path.basename(self.mask_img))
        
        # check if the right number of seed is provided for the statistical analysis
        for seed_name in self.seed_names:
            if self.model=="OneSampleT" and len(self.seed_names)!=1:
                raise ValueError(">>>> Only One seed should be provided for One sample t-test or try 'TwoSampT_paired' or 'TwoSampT_unpaired' or 'HigherOrder_paired'")
            elif self.model=="TwoSampT_paired" and len(self.seed_names)!=2 or self.model=="TwoSampT_unpaired" and len(self.seed_names)!=2:
                raise ValueError(">>>> Two seeds should be provided for Two sample t-test or try 'OneSampleT' or 'HigherOrder_paired'")


                
        #>>> II. Create output directory (if save_ana=True) -------------------------------------
        if save_ana==True:
            self.output_dir=self.config['main_dir'] + self.config['seed2vox_dir'] + '/2_second_level/'+self.model+'/'+ os.path.basename(self.mask_img).split(".")[0] +'/' +self.measure +"/" + self.output_tag # name of the output dir
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir,exist_ok=True)  # Create output directory:
            print("> Saved here : " + self.output_dir)
            print("  ")
                

        #>>> III. Select first level data: -------------------------------------
        self.data_1rstlevel={};
        for seed_name in self.seed_names:
            self.data_1rstlevel[seed_name]=[]
            for sbj_nb in range(len(config['list_subjects'])):
                subject_name='sub-' +  config['list_subjects'][sbj_nb]
                tag_files = {"Corr": "corr","MI": "mi"}
                self.tag_file = tag_files.get(self.measure, None)
                self.data_1rstlevel[seed_name].append(glob.glob(self.config["first_level"] +'/'+ seed_name+'/'+ self.target +'_fc_maps/'+ self.measure + "/" +self.tag_file + "_"+ subject_name + "*.nii.gz")[0])

                    
    def design_matrix(self,contrast_name=None,plot_matrix= False,save_matrix=False):
        '''
        Create and plot the design matrix for "OneSampleT" or  "TwoSampT_unpaired" or "TwoSampT_paired"
        For one matrix per contrast is created
        
        Attributes
        ----------
        contrast_name: str 
        
        plot_matrix: bool (default: False)
            To plot the design matrix
        
        save_matrix: default: False
            To save the design matrix. If True then plot_matrix will be turn True
            output: 
                - design_matrix_*.png : Save the image of the desgin matrix
                design_matrix_*.npy : Save the desgin matrix in numpy format
                
        
        To add: Contrats: dict (default : None)
            it will not be used for "OneSampleT" or  "TwoSampT_unpaired" or "TwoSampT_paired" models
            
            A dictonnary should be provided for complexe models "HigherOrder_paired"
            Contrast={"nameofthecontrast1": contrast1_values,
            "nameofthecontrast2": contrast2_values}
            contrast_values shape: (number of functional images), exemple: [1,1,0,0] for 4 functional images
        
        Return
        ----------
        Design_matrix:  dict
        Contained one matrix for each contrast (1: matrix for OnSample T and 4 matrices for TwoSampT )
        '''
        
        #>>>>>>>>  Initiate variables
        self.contrast_name=contrast_name
        if save_matrix==True:
            plot_matrix==True # matrix cannot be saved if them has not been plot
        
        n_subjects=len(self.config["list_subjects"]) # indicate the totalnumber of participant to analyse
        contrasts={}; # create empty dict
        Design_matrix={} # create empty dict
                    
        #>>>>>>>> Create contrasts for each kind of test ____________________
        contrasts=self._generate_contrast()
        
        #>>>>>>>> Create a desgin matrix for each kind of test ____________________
        # For un unpaired tests:
        if self.model=="OneSampleT" or self.model=="TwoSampT_unpaired" :
            for i, (contrast,values) in enumerate(contrasts.items()):
                Design_matrix[contrast]=pd.DataFrame(np.hstack((values[:, np.newaxis])), columns=[contrast])
            
        # For paired tests:                            
        elif self.model=="TwoSampT_paired" or self.model=="HigherOrder_paired":
            contrasts=self._generate_contrast()
            subjects = self.config["list_subjects"]
            
            # Add subject effect:
            for i, (contrast,values) in enumerate(contrasts.items()):
                Design_matrix[contrast]=pd.DataFrame(np.hstack((values[:, np.newaxis],np.concatenate([np.eye(n_subjects)] * len(self.seed_names), axis=0))), columns=[contrast] + subjects)
            

        #Plot the matrix
        ### Create a subplot per contrast
        if plot_matrix== True:
            if self.model=="OneSampleT" or self.model=="TwoSampT_unpaired" :
                num_subplots = len(Design_matrix)# Define the number of subplots and their layout
                fig, axes = plt.subplots(1, num_subplots ,figsize=(len(Design_matrix)*2,5)) # # Create a figure and subplots
            
            elif self.model=="TwoSampT_paired" or self.model=="HigherOrder_paired":
                if len(self.seed_names) <5:
                    num_subplots = int(len(Design_matrix)) # Define the number of subplots and their layout
                    fig, axes = plt.subplots(1, num_subplots ,figsize=(len(Design_matrix)*4,5)) # # Create a figure and subplots
                else:  
                    num_subplots = int(math.ceil(len(Design_matrix)/2)) # Define the number of subplots and their layout
                    fig, axes = plt.subplots(2, num_subplots ,figsize=(len(Design_matrix)*3,10)) # # Create a figure and subplots

            if num_subplots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
                
            # Loop over the subplots and plot the data
            for i, (title,values) in enumerate(Design_matrix.items()):
                plot_design_matrix(values,ax=axes[i])
                axes[i].set_ylabel("maps")
            fig.suptitle("Desgin matrix " + self.model,fontsize=12)
            plt.tight_layout() # Adjust the layout and spacing of subplots
            
            
            if save_matrix==True:
                if not os.path.exists(self.output_dir):
                     raise ValueError(">>>> " +self.output_dir+ " directory should be created first with Stats() function")
                
                plt.savefig(self.output_dir + "/design_matrix_" + self.output_tag + ".png")
                np.save(self.output_dir + "/design_matrix_" + self.output_tag + ".npy",Design_matrix)
                
            plt.show() # Display the figure
                
        return Design_matrix
        
    def secondlevelmodel(self,Design_matrix,parametric=True,plot_2ndlevel=False,save_img=False):
        '''
        This function calculate the second level model
        
        Attributes
        ----------
        Design_matrix: Dict
            The information about the contrasts. Contained one matrix for each contrast
            
        plot_2ndlevel: bool (default: False)
            To plot the uncorrected zmaps for each contrast
        
        save_img: bool, default: False
            To save the uncorrected maps for each contrast and each stats (.nii.gz)
                
        https://nilearn.github.io/dev/modules/generated/nilearn.glm.compute_contrast.html
        
        Return
        ----------
        contrast_map:  nifti images
        - 'z_score': z-maps
        - 'stat: t-maps
        - 'p_value': p-values maps
        - 'effect_size'
        - 'effect_variance'

        '''
        self.parametric=parametric
        # concatenates the files if there are multiple factors:
        input_files=[]
        for seed_name in self.seed_names:
            if len(self.seed_names)==1:
                input_files=self.data_1rstlevel[seed_name]
            elif len(self.seed_names)>1:
                input_files=np.concatenate((input_files,self.data_1rstlevel[seed_name]))
        
            
        # Load the nifti files
        nifti_files=[]
        for i in range(0,len(input_files)):
            nifti_files.append(nb.load(input_files[i]))
            
        # fit the model for each matrix and compute the constrast for parametrical statistics
       
        second_level_model={};contrast_map={}
        for i, (title,values) in enumerate(Design_matrix.items()):
            if parametric==True:
                second_level_model[title] = SecondLevelModel(mask_img=self.mask_img,smoothing_fwhm=None)

                second_level_model[title] = second_level_model[title].fit(nifti_files, design_matrix=Design_matrix[title])
                contrast_map[title]=second_level_model[title].compute_contrast(title, second_level_stat_type="t",output_type="all")
                if save_img==True:
                    output_uncorr=self.output_dir + "/uncorr/"
                    if not os.path.exists(output_uncorr):
                        os.mkdir(output_uncorr)
                    nb.save(contrast_map[title]['z_score'], output_uncorr +"/zscore_" + title + ".nii.gz")
                    nb.save(contrast_map[title]['stat'],output_uncorr + "/stat_" + title + ".nii.gz") # t or F statistical value
                    nb.save(contrast_map[title]['p_value'],output_uncorr + "/pvalue_" + title + ".nii.gz")
                    nb.save(contrast_map[title]['effect_size'],output_uncorr + "/effectsize_" + title + ".nii.gz")
                    nb.save(contrast_map[title]['effect_variance'],output_uncorr+ "/effectvar_" + title + ".nii.gz")
                
                if plot_2ndlevel==True:
                        plotting.plot_glass_brain(
                            contrast_map[title]['stat'],
                            colorbar=True,
                            symmetric_cbar=False,
                            display_mode='lyrz',
                            threshold=2.5,
                            vmax=5,
                            title=title)

                    
            elif parametric==False:
                contrast_map[title]=non_parametric_inference(nifti_files,design_matrix=Design_matrix[title],model_intercept=True,n_perm=50, # should be set between 1000 and 10000
                                         two_sided_test=False,
                                         mask=None,
                                         smoothing_fwhm=None,
                                         tfce=False, # choose tfce=True or threshold is not None
                                         threshold=0.01,
                                         n_jobs=8)
                if save_img==True:
                    output_uncorr=self.output_dir + "/nonparam/"
                    if not os.path.exists(output_uncorr):
                        os.mkdir(output_uncorr)
                    nb.save(contrast_map[title]['t'], output_uncorr +"/t_" + title + ".nii.gz")
                    nb.save(contrast_map[title]['size'], output_uncorr +"/size_" + title + ".nii.gz")
                    nb.save(contrast_map[title]['logp_max_t'],output_uncorr + "/logp_max_t_" + title + ".nii.gz") # t or F statistical value
                    nb.save(contrast_map[title]['logp_max_size'],output_uncorr + "/logp_max_size_" + title + ".nii.gz")
                    nb.save(contrast_map[title]['mass'],output_uncorr + "/mass_" + title + ".nii.gz")
                    nb.save(contrast_map[title]['logp_max_mass'],output_uncorr+ "/logp_max_mass_" + title + ".nii.gz")
                    #nb.save(contrast_map[title]['tfce'],output_uncorr + "/tfce_" + title + ".nii.gz")
                    #nb.save(contrast_map[title]['logp_max_tfce'],output_uncorr+ "/logp_max_tfce_" + title + ".nii.gz")
            
                if plot_2ndlevel==True:
                    plotting.plot_glass_brain(
                            contrast_map[title]['size'],
                            colorbar=True,
                            symmetric_cbar=False,
                            display_mode='lyrz',
                            threshold=2.5,
                            vmax=5,
                            title=title)

                       
        # fit the model for each matrix and compute the constrast for non-parametrical statistics
   
        return contrast_map
    
    def secondlevel_correction(self,maps,z_thr=1.5,p_value=0.001,cluster_threshold=10,corr=None,smoothing=None,plot_stats_corr=False,save_img=False,n_job=1):
        '''
        One sample t-test
        Attributes
        ----------
        maps: statisitcal maps from 2nd level fitting
        
        p_value : float or list, optional
        Number controlling the thresholding (either a p-value or q-value). Its actual meaning depends on the height_control parameter. This function translates p_value to a z-scale threshold. Default=0.001.

        corr: string, or None optional
        False positive control meaning of cluster forming threshold: None|’fpr’|’fdr’|’bonferroni’ Default=’fpr’.

        '''
    
        for i, (title,values) in enumerate(maps.items()):
            if self.parametric== True:

                thresholded_map, threshold = threshold_stats_img(maps[title]["z_score"],alpha=p_value,threshold=z_thr,height_control=corr,cluster_threshold=cluster_threshold,two_sided=False)
            

            #TO DO:
                    #else:
                    #non_parametric_inference(second_level_input,
               #     design_matrix=design_matrix,
                #    model_intercept=True,
                 #   n_perm=50, # should be set between 1000 and 10000
                  #  two_sided_test=False,
                  #  mask=None,
                   # smoothing_fwhm=None,
                    #tfce=True, # choose tfce=True or threshold is not None
                    #threshold=p_value,
                   # n_jobs=n_job)

                #thresholded_map=img_dict["logp_max_t"] ; threshold=1; cluster_threshold=0#img_dict["tfce"]




            if plot_stats_corr==True:
                thresholded_map=image.threshold_img(thresholded_map, 0,mask_img=self.mask_img, copy=True)

                self._plot_stats(thresholded_map, title ,threshold, cluster_threshold)

            if save_img==True:
                output_corrected=self.output_dir + "/"+corr+"_corrected/"
                #print(output_corrected + "/" + title + "_" +corr+ "_p"+ str(p_value).split('.')[-1] +".nii.gz")
                if not os.path.exists(output_corrected):
                    os.mkdir(output_corrected)
                nb.save(thresholded_map, output_corrected + "/" + title + "_" +corr+ "_p"+ str(p_value).split('.')[-1] +".nii.gz")
                
        
        
    def _plot_stats(self,second_level_map,title,threshold,cluster_threshold):
        '''
        Extract significant cluster in a table
        Attributes
        ----------
        stat_img : Niimg-like object
        Statistical image to threshold and summarize.

        stat_threshold float
        Cluster forming threshold. This value must be in the same scale as stat_img.

        '''
        display = plotting.plot_glass_brain(
            second_level_map,
            colorbar=True,

            display_mode='lyrz',
            title=title)  
        plotting.show()
        
    
    def _cluster_table(stat_img,stat_threshold):
        '''
        Extract significant cluster in a table
        Attributes
        ----------
        stat_img : Niimg-like object
        Statistical image to threshold and summarize.

        stat_threshold float
        Cluster forming threshold. This value must be in the same scale as stat_img.

        '''

        table = get_clusters_table(z_map, stat_threshold, cluster_threshold=10)
        return table
        
    def _generate_contrast(self):
        
        contrast_name=self.contrast_name
        contrasts={}
        
        if self.model== "OneSampleT" and contrast_name==None: 
            contrasts["Main " + self.seed_names[0]]=np.hstack(([1] * len(self.config['list_subjects'])))
            
        elif self.model=="TwoSampT_paired" or self.model=="TwoSampT_unpaired" :
            contrasts["Main " + self.seed_names[0]]=np.hstack(([1] * len(self.config['list_subjects']), [0] * len(self.config['list_subjects']))) # Main contrast for the first factor
            contrasts["Main " + self.seed_names[1]]=np.hstack(([0] * len(self.config['list_subjects']), [1] * len(self.config['list_subjects']))) # Main contrast for the second factor
            contrasts[self.seed_names[0] + " vs " + self.seed_names[1]]=np.hstack(([1] * len(self.config['list_subjects']), [-1] * len(self.config['list_subjects']))) # contrast between the two
            contrasts[self.seed_names[1] + " vs " + self.seed_names[0]]=np.hstack(([-1] * len(self.config['list_subjects']), [1] * len(self.config['list_subjects']))) # contrast between the two
            contrasts["All effect"]=np.hstack(([1] * len(self.config['list_subjects']), [1] * len(self.config['list_subjects'])))
        
        
        elif len(self.seed_names)==4 and contrast_name==None:
            #VR, VL, DR, DL
            for seed_nb in range(0,len(self.seed_names)):
                contrasts["Main test " + self.seed_names[seed_nb]]=np.hstack(([0] * len(self.config['list_subjects']) * seed_nb, [1] * len(self.config['list_subjects']), [0] * len(self.config['list_subjects'])* (len(self.seed_names)-(seed_nb+1))))
            contrasts["Ventral Effect"]=np.hstack(([1] * len(self.config['list_subjects']) *2, [0] * len(self.config['list_subjects']) * 2))
            contrasts["Dorsal Effect"]=np.hstack(([0] * len(self.config['list_subjects'])*2, [1] * len(self.config['list_subjects']) * 2))
            contrasts["Right Effect"]=np.hstack((np.tile(([[1] * len(self.config['list_subjects'])+ [0] * len(self.config['list_subjects'])]),2)))
            contrasts["Left Effect"]=np.hstack((np.tile(([[0] * len(self.config['list_subjects'])+ [1] * len(self.config['list_subjects'])]),2)))
            contrasts["Ventral vs Dorsal"]=np.hstack(([1] * len(self.config['list_subjects']) *2, [-1] * len(self.config['list_subjects']) * 2))
            contrasts["Dorsal vs Ventral"]=np.hstack(([-1] * len(self.config['list_subjects'])*2, [1] * len(self.config['list_subjects']) * 2))
            contrasts["Right vs Left"]=np.hstack((np.tile(([[1] * len(self.config['list_subjects'])+ [-1] * len(self.config['list_subjects'])]),2)))
            contrasts["Left vs Righ"]=np.hstack((np.tile(([[-1] * len(self.config['list_subjects'])+ [1] * len(self.config['list_subjects'])]),2)))
            contrasts["All effect"]=np.hstack(([1] * len(self.config['list_subjects']) * len(self.seed_names)))
                    
        elif len(self.seed_names)>4 and contrast_name==None:
            for seed_nb in range(0,len(self.seed_names)):
                contrasts["Main test " + self.seed_names[seed_nb]]=np.hstack(([0] * len(self.config['list_subjects']) * seed_nb, [1] * len(self.config['list_subjects']), [0] * len(self.config['list_subjects'])* (len(self.seed_names)-(seed_nb+1))))
            contrasts["All effect"]=np.hstack(([1] * len(self.config['list_subjects']) * len(self.seed_names)))
            
        elif contrast_name=="4quad_9levels":
            #for seed_nb in range(0,len(self.seed_names)):
                #contrasts["Main test " + self.seed_names[seed_nb]]=np.hstack(([0] * len(self.config['list_subjects']) * seed_nb, [1] * len(self.config['list_subjects']), [0] * len(self.config['list_subjects'])* (len(self.seed_names)-(seed_nb+1))))
            contrasts["Ventral Effect"]=np.hstack(([1] * len(self.config['list_subjects']) *9*2, [0] * len(self.config['list_subjects']) * 9*2))
            contrasts["Dorsal Effect"]=np.hstack(([0] * len(self.config['list_subjects'])*9*2, [1] * len(self.config['list_subjects']) * 9*2))
            contrasts["Right Effect"]=np.hstack((np.tile(([[1] * len(self.config['list_subjects'])+ [0] * len(self.config['list_subjects'])]),9*2)))
            contrasts["Left Effect"]=np.hstack((np.tile(([[0] * len(self.config['list_subjects'])+ [1] * len(self.config['list_subjects'])]),9*2)))
            contrasts["Ventral vs Dorsal"]=np.hstack(([1] * len(self.config['list_subjects']) *9*2, [-1] * len(self.config['list_subjects']) * 9*2))
            contrasts["Dorsal vs Ventral"]=np.hstack(([-1] * len(self.config['list_subjects'])*9*2, [1] * len(self.config['list_subjects']) * 9*2))
            contrasts["Right vs Left"]=np.hstack((np.tile(([[1] * len(self.config['list_subjects'])+ [-1] * len(self.config['list_subjects'])]),9*2)))
            contrasts["Left vs Right"]=np.hstack((np.tile(([[-1] * len(self.config['list_subjects'])+ [1] * len(self.config['list_subjects'])]),9*2)))
            contrasts["All effect"]=np.hstack(([1] * len(self.config['list_subjects']) * len(self.seed_names)))
            
        elif contrast_name=="D-R_9levels":
            for seed_nb in range(0,len(self.seed_names)):
                contrasts["Main test " + self.seed_names[seed_nb]]=np.hstack(([0] * len(self.config['list_subjects']) * seed_nb, [1] * len(self.config['list_subjects']), [0] * len(self.config['list_subjects'])* (len(self.seed_names)-(seed_nb+1))))
            contrasts["Right Effect"]=np.hstack((np.tile(([[1] * len(self.config['list_subjects'])+ [0] * len(self.config['list_subjects'])]),9)))
            contrasts["Left Effect"]=np.hstack((np.tile(([[0] * len(self.config['list_subjects'])+ [1] * len(self.config['list_subjects'])]),9)))
            contrasts["Right vs Left"]=np.hstack((np.tile(([[1] * len(self.config['list_subjects'])+ [-1] * len(self.config['list_subjects'])]),9)))
            contrasts["Left vs Right"]=np.hstack((np.tile(([[-1] * len(self.config['list_subjects'])+ [1] * len(self.config['list_subjects'])]),9)))
            contrasts["All effect"]=np.hstack(([1] * len(self.config['list_subjects']) * len(self.seed_names)))
                    
        
        
        return contrasts
