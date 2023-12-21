# -*- coding: utf-8 -*-
import os, glob, shutil

class Surface:
    '''
    The Seed2voxels class is used to run correlation analysis
    Attributes
    ----------
    config : dict
    '''
    
    def __init__(self, config,ana_level="group",verbose=True):
        self.config = config # load config info
        self.participant_ids= config["list_subjects"]
        self.preprocess_dir=[]
        self.output_dir= config["main_dir"]+config["analysis_dir"]
        self.ana_level=ana_level
        
        
        if self.ana_level=="indiv":
            for participant_nb in range(0,len(self.participant_ids)):
                if self.config["list_dataset"][participant_nb]=="bmpd":
                    self.preprocess_dir.append(config["prepross_bmpd_dir"])
                else:
                    self.preprocess_dir.append(config["prepross_stratals_dir"])



            #>>> create output directory if needed -------------------------------------
            if not os.path.exists(self.output_dir + '/1_first_level/'):
                    os.mkdir(self.output_dir + '/1_first_level/')
            
            #>>> Select data: -------------------------------------
            self.anat_files=[];
            for participant_nb in range(0,len(self.participant_ids)):
                self.anat_files.append(glob.glob(self.preprocess_dir[participant_nb] + "sub-" + self.participant_ids[participant_nb] + "/" + self.config["anat"]["coreg_dir"] + self.config["anat"]["coreg_file"])[0])

            if verbose==True:
                print("ready to run surface analyses on " + str(len(self.participant_ids)) + " participants")
    
    
        elif ana_level=="group":
             #>>> create output directory if needed -------------------------------------
            if not os.path.exists(self.output_dir + '/group_level/'):
                        os.mkdir(self.output_dir + '/group_level/')
            
            self.group_file=glob.glob(config["main_dir"] + self.config["anat"]["group_mean"])[0]
            
            if verbose==True:
                print("ready to run surface analyses on: ")
                print(self.group_file)

             
                

    def preprocess_anat(self,redo=False):
        '''
        Use recon-all from freesurfer
        details here: https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all
       
        
        Inputs
        ----------
        
        Returns
        ----------
        
            
        '''
        if self.ana_level=="indiv":
            for participant_nb in range(0,len(self.participant_ids)):
                participant_id=self.participant_ids[participant_nb] + "_bis"
                print(">>>>> Surface analysis is running for participant: " + participant_id)
                anat_file=self.anat_files[participant_nb]
                output_dir=self.output_dir + '/1_first_level/' 
                string="recon-all -all -subjid " + participant_id + " -i " + anat_file + " -sd " + output_dir
                
                os.system(string) 
                print("done")
                print("  ")
        
        elif self.ana_level=="group":
            output_dir=self.output_dir + '/group_level/' 
            ana_name="group_n" + str(len(self.participant_ids))
            
            
            if not os.path.exists(output_dir + ana_name +"/mri") or redo==True:
                print(">>>>> Surface analysis is running on mean group image")
                string1="recon-all -autorecon2 -subjid " + ana_name + " -sd " + output_dir + " -no-isrunning "
                string2="recon-all -autorecon-pial -subjid " + ana_name + " -sd " + output_dir + " -no-isrunning "
                 
                os.system(string1);os.system(string2)
                print("done")
                print("  ")
                
            else:
                print(">>>>> Surface analysis was already done, set redo=True to redo the analysis")
        


    def vol2surf(self,vol_file,output_file,redo=False):
        '''
        Use vol2surf from freesurfer
        details here: https://surfer.nmr.mgh.harvard.edu/fswiki/mri_vol2surf
        
        Inputs
        ----------
        vol_file: str
            input filename (volume)
        
        output_file: str
            outpur filename (surface)
        
        
        Returns
        ----------
        
            
        '''
        anat_dir=self.output_dir + '/group_level/' 
        target_name="/group_n" + str(len(self.participant_ids)) 
        string1="mri_vol2surf --src "+vol_file + ".nii.gz --out "+output_file+ "lh." + os.path.basename(vol_file) + ".mgh"+" --regheader fsaverage --hemi lh " 
        string2="mri_vol2surf --src "+vol_file + ".nii.gz --out "+output_file+ "rh." + os.path.basename(vol_file) + ".mgh"+" --regheader fsaverage --hemi rh " 

        os.system(string1);os.system(string1)