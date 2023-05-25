# -*- coding: utf-8 -*-
import glob, os, shutil, json, scipy
#from urllib.parse import non_hierarchical
import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn import image
from joblib import Parallel, delayed

import dcor 
import pingouin as pg
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn import decomposition
from scipy import stats

from tqdm import tqdm
# to improve---------------------------
# loop per seed

# extract fct: 
# plotting
#

class Seed2voxels:
    '''
    The Seed2voxels class is used to run correlation analysis
    Attributes
    ----------
    config : dict
    signal: str
        type of signal ('raw' for bold or 'ai' for deconvoluted)
    '''
    
    def __init__(self, config, signal):
        self.config = config # load config info
        self.signal=signal # define if the signal is raw or ai
        self.subject_names= config["list_subjects"]
        self.outputdir= self.config["main_dir"] +self.config["seed2vox_dir"]
        self.seed_names=self.config["seeds"]["seed_names"]
        self.target=self.config["targeted_voxels"]["target_name"]
        self.seed_structure=self.config["seeds"]["seed_structure"]
        self.target_structure=self.config["targeted_voxels"]["target_structure"]
        
        #>>> create output directory if needed -------------------------------------
        if not os.path.exists(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+self.target):
                os.mkdir(self.config['main_dir'] + self.config['seed2vox_dir'] + '/1_first_level/'+self.target)
                os.mkdir(self.config['main_dir'] + self.config['seed2vox_dir'] + '/1_first_level/'+self.target+'/timeseries/') # folder to store timeseries extraction
                os.mkdir(self.config['main_dir'] + self.config['seed2vox_dir'] + '/1_first_level/'+self.target+'/' + self.target +'_fc_maps/') # folder to store maps of FC

                
        for seed_name in self.seed_names:
            if not os.path.exists(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name):
                os.mkdir(self.config['main_dir'] + self.config['seed2vox_dir'] + '/1_first_level/'+seed_name)
                os.mkdir(self.config['main_dir'] + self.config['seed2vox_dir'] + '/1_first_level/'+seed_name+'/timeseries/') # folder to store timeseries extraction
                os.mkdir(self.config['main_dir'] + self.config['seed2vox_dir'] + '/1_first_level/'+seed_name+'/'+ self.target +'_fc_maps/') # folder to store maps of FC
        

    
        #>>> Select mask: -------------------------------------
        self.mask_target=glob.glob(self.config["main_dir"] + self.config["targeted_voxels"]["target_dir"]+ self.target + ".nii.gz")[0] # mask of the voxels tareted for the analysis
        print("Start the analysis on: " + str(len(self.subject_names))+ " participants")
        print("targeted voxel's mask: " + self.target)
        #print(self.mask_target)
        
        self.mask_seeds={}
        for seed_name in self.seed_names:
            print(self.config["main_dir"] + self.config["seeds"]["seed_dir"]+ seed_name + ".nii.gz")
            self.mask_seeds[seed_name]=glob.glob(self.config["main_dir"] + self.config["seeds"]["seed_dir"]+ seed_name + ".nii.gz")[0] # mask of the voxels tareted for the analysis
            #print(self.mask_seeds[seed_name])
        
        #>>> Select data: -------------------------------------
        self.data_seed=[];self.data_target=[]; 
        for subject_name in config['list_subjects']:
            subject_name='sub-' +  subject_name
            
            # images selected for extraction:
            self.data_seed.append(glob.glob(self.config["input_func"]["seed_dir"] + subject_name +'/'+ self.seed_structure +'/*'+ config["input_func"]["seed_tag"] +'*')[0])
           
            self.data_target.append(glob.glob(self.config["input_func"]["target_dir"] + subject_name +'/'+ self.target_structure +'/*'+ config["input_func"]["target_tag"] +'*')[0])
             
                
        
    def extract_data(self,smoothing_seed=None,smoothing_target=None,redo=False,n_jobs=0):
        
        '''
        Extracts time series in a mask or load them from text files
        At least 2 masks should be provided 1 for the target and one for the seed.
        More than one mask could be provide for the seeds.
        
        Inputs
        ----------
        smoothing_fwhm: array
            to apply smoothing during the extraction (ex: [6,6,6])
        redo: 
            if True the extraction will be rerun else if the timeseries were already extracted, the file conaining the data will be load
        n_jobs: "extract", "load"
             
        Returns
        ----------
        timeseries_target:
            timeseries of each voxels for each participant
        timeseries_target_mean:
            mean timeserie in the mask for each participant
        timeseries_target_pc1:
            principal component in the mask for each participant
            
        timeseries_seeds:
            timeseries of each voxels for each participant
        timeseries_seeds_mean:
            mean timeserie in the mask for each participant
        timeseries_seeds_pc1:
            principal component in the mask for each participant
            
        '''
        ts_target_dir=[]; ts_target_txt=[];
        ts_seeds_dir={};ts_seeds_txt={};
        
        timeseries_target={"raw":[],"zscored":[],"mean":[],"zmean":[],"PC1":[]}
        timeseries_seeds={"raw":{},"zscored":{},"mean":{},"zmean":{},"PC1":{}}

    # 1. Define Output filename (timeseries) _____________________________________
        ts_target_dir=self.config['main_dir'] + self.config['seed2vox_dir'] + '/1_first_level/'+self.target+'/timeseries/' # output diretory for targeted voxel's mask
        for seed_name in self.seed_names:
            ts_seeds_dir[seed_name]=self.config['main_dir'] + self.config['seed2vox_dir'] + '/1_first_level/'+seed_name+'/timeseries/' # output diretory for seeds mask
           
        for subject_name in self.subject_names:
            ts_target_txt.append(ts_target_dir + '/sub_' + subject_name + '_mask_' + self.target + '_timeseries') # output file for targeted voxel's mask
        
        for seed_name in self.seed_names:
            ts_seeds_txt[seed_name]=[]
            for subject_name in self.subject_names:
                ts_seeds_txt[seed_name].append(ts_seeds_dir[seed_name] + '/sub_' + subject_name + '_mask_' + seed_name + '_timeseries') # output file for targeted voxel's mask
                
    # 2. Extract signal (timeseries) _____________________________________
    # Signals are extracted differently for ica and icaps

        if self.signal == 'raw':
            
            ## a. Extract or load data in the targeted voxel mask ___________________________________________

            
            ts_target=Parallel(n_jobs=n_jobs)(delayed(self._extract_ts)(self.mask_target,self.data_target[subject_nb],ts_target_txt[subject_nb],redo,smoothing_target)
                                        for subject_nb in range(len(self.subject_names)))
            
            with open(os.path.dirname(ts_target_txt[0]) + '/seed2voxels_analysis_config.json', 'w') as fp:
                json.dump(self.config, fp)

            for subject_nb in range(len(self.subject_names)):
                timeseries_target["raw"].append(ts_target[subject_nb][0]); timeseries_target["zscored"].append(ts_target[subject_nb][1]);
                timeseries_target["mean"].append(ts_target[subject_nb][2]); timeseries_target["zmean"].append(ts_target[subject_nb][2]);
                timeseries_target["PC1"].append(ts_target[subject_nb][3]);

            ## 2. Extract data in seeds ___________________________________________
            for seed_nb in tqdm(range(0,len(self.seed_names)), desc ="data extracted"):
                seed_name=self.seed_names[seed_nb]
                timeseries_seeds["raw"][seed_name]=[]; timeseries_seeds["zscored"][seed_name]=[]; timeseries_seeds["mean"][seed_name]=[]; timeseries_seeds["zmean"][seed_name]=[];
                timeseries_seeds["PC1"][seed_name]=[];
                

                ts_seeds=Parallel(n_jobs=n_jobs)(delayed(self._extract_ts)(self.mask_seeds[seed_name],self.data_seed[subject_nb],ts_seeds_txt[seed_name][subject_nb],smoothing_seed)
                                            for subject_nb in range(len(self.subject_names)))
                
                with open(os.path.dirname(ts_seeds_txt[seed_name][0]) + '/seed2voxels_analysis_config.json', 'w') as fp:
                    json.dump(self.config, fp)

                for subject_nb in range(len(self.subject_names)):
                    timeseries_seeds["raw"][seed_name].append(ts_seeds[subject_nb][0])
                    timeseries_seeds["zscored"][seed_name].append(ts_seeds[subject_nb][1])
                    timeseries_seeds["mean"][seed_name].append(ts_seeds[subject_nb][2])
                    timeseries_seeds["zmean"][seed_name].append(ts_seeds[subject_nb][3])
                    timeseries_seeds["PC1"][seed_name].append(ts_seeds[subject_nb][4])

            
        elif self.signal == 'ai':
            print("Warning: that part of the script should be update")
            if run == "load":
                timeseries=Parallel(n_jobs=n_jobs)(delayed(np.loadtxt)(timeseries_txt[subject_nb] + '.txt') for subject_nb in range(len(self.subject_names)))

            elif run == "extract":
                print('Note: when using "ai", signals can only be extracted for targets.')
                timeseries = Parallel(n_jobs=n_jobs)(delayed(self._extract_ts)(mask[subject_nb],img[subject_nb],timeseries_txt[subject_nb],smoothing_fwhm)
                                            for subject_nb in range(len(self.subject_names)))

                with open(os.path.dirname(timeseries_txt[0]) + '/seed2voxels_analysis_config.json', 'w') as fp:
                        json.dump(self.config, fp)
        
        print("Ouptuts have the following contain:")
        print("1: timeseries_target={'raw':[],'zscored':[],'mean':[],'zmean':[],'PC1':[]}")
        print("2: timeseries_seeds={'raw':[],'zscored':[],'mean':[],'zmean':[],'PC1':[]}")
        
        return timeseries if self.signal=='ai' else (timeseries_target,timeseries_seeds)


    def _extract_ts(self,mask,img,ts_txt,redo,smoothing=None):
        '''
        Extracts time series in a mask + calculates the mean and PC
        '''
        
        if not os.path.exists(ts_txt + '.npy') or redo==True:
            masker= NiftiMasker(mask,smoothing_fwhm=smoothing, t_r=1.55,low_pass=None, high_pass=None if self.signal == 'raw' else None) # seed masker
            ts=masker.fit_transform(img) #low_pass=0.1,high_pass=0.01
            np.savetxt(ts_txt + '.npy',ts)

            # Calculate the z-scored time serie
            ts_zscored=stats.zscore(ts,axis=0) # mean time serie

            np.savetxt(ts_txt + '_zscored.npy',ts_zscored)
            
            # Calculate the mean time serie
            ts_mean=np.mean(ts,axis=1) # mean time serie
            np.savetxt(ts_txt + '_mean.npy',ts_mean)
            
            # Calculate the mean time serie
            ts_zmean=np.mean(ts_zscored,axis=1) # mean time serie
            np.savetxt(ts_txt + '_zmean.npy',ts_zmean)
            
            # Calculate the principal component:
            pca=decomposition.PCA(n_components=1)
            pca_components=pca.fit_transform(ts)
            ts_pc1=pca_components[:,0] 
            np.savetxt(ts_txt + '_PC1.npy',ts_pc1)
            
            
        
        else:
            ts=np.loadtxt(ts_txt + '.npy')
            ts_zscored=np.loadtxt(ts_txt + '_zscored.npy')
            ts_zmean=np.loadtxt(ts_txt + '_zmean.npy')
            ts_mean=np.loadtxt(ts_txt + '_mean.npy')
            ts_pc1=np.loadtxt(ts_txt + '_PC1.npy')
             
       
        
        return ts if self.signal=='ai' else (ts,ts_zscored, ts_mean, ts_zmean, ts_pc1)

    def correlation_maps(self,seed_ts,voxels_ts,output_img=None,Fisher=True,partial=False,save_maps=True,smoothing_output=None,redo=False,n_jobs=1):
        '''
        Compute correlation maps between a seed timecourse and voxels
        Inputs
        ----------
        seed_ts: list
            timecourse to use as seed (see extract_data method) (list containing one array per suject)
        target_ts: list
            timecourses of all voxels on which to compute the correlation (see extract_data method) (list containing one array per suject)
        output_img: str
            path + rootname of the output image (/!\ no extension needed) (ex: '/pathtofile/output')
        Fisher: boolean
            to Fisher-transform the correlation (default = True).
        partial:
            Run partial correlation i.e remove the first derivative on the target signal (default = False).
        save_maps: boolean
            to save correlation maps (default = True).
        redo: boolean
            to rerun the analysis
        njobs: int
            number of jobs for parallelization
    
        Output
        ----------
        output_img_zcorr.nii (if Fisher = True) or output_img_corr.nii (if Fisher = False) 
            4D image containing the correlation maps for all subjects
        subject_labels.txt
            text file containing labels of the subjects included in the correlation analysis
        correlations: array
            correlation maps as an array
        '''

        if not os.path.exists(output_img) or redo==True:
            
            correlations = Parallel(n_jobs=n_jobs)(delayed(self._compute_correlation)(subject_nb,
                                                                                          seed_ts[subject_nb],
                                                                                          voxels_ts[subject_nb],
                                                                                          output_img,
                                                                                          Fisher,
                                                                                          partial,
                                                                                          save_maps)
                                           for subject_nb in range(len(self.subject_names)))


            if save_maps==True:
                Parallel(n_jobs=n_jobs)(delayed(self._save_maps)(subject_nb,
                                                                 correlations[subject_nb],
                                                                 output_img,
                                                                 smoothing_output)
                                           for subject_nb in range(len(self.subject_names)))



            #for tmp in glob.glob(os.path.dirname(output_img) + '/tmp_*.nii.gz'):
               # os.remove(tmp) # remove temporary 3D images files

            np.savetxt(os.path.dirname(output_img) + '/subjects_labels.txt',self.subject_names,fmt="%s") # copy the config file that store subject info


            
        else:
            print("The correlation maps were alredy computed please put redo=True to rerun the analysis")
    
        return correlations 
    
    def _compute_correlation(self,subject_nb,seed_ts,voxels_ts,output_img,Fisher,partial,save_maps):
        '''
        Run the correlation analyses.
        The correlation can be Fisher transformed (Fisher == True) or not  (Fisher == False)
        The correlation could be classical correlations (partial==False) or partial correlations (partial==True)
        For the last option:
        > 1. we calculated the first derivative of the signal (in each voxel of the target)
        > 2. the derivative is used as a covariate in pg.partial_corr meaning that the derivative is remove for the target but no seed signal (semi-partial correlation)
        ----------
       '''
        
        seed_to_voxel_correlations = np.zeros((voxels_ts.shape[1], 1)) # np.zeros(number of voxels,1)
        if partial==False: # compute correlation
            for v in range(0,voxels_ts.shape[1]): 
                seed_to_voxel_correlations[v] = np.corrcoef(seed_ts, voxels_ts[:, v])[0, 1]
                
        elif partial==True: # compute correlation
            target_derivative = np.zeros((voxels_ts.shape[0]-1, voxels_ts.shape[1])) # np.zeros(number of voxels,1)
            for v in range(0,voxels_ts.shape[1]-1): 
                target_derivative[:,v] = np.diff(voxels_ts[:, v]) # calculate the first derivative of the signal
                df={'seed_ts':seed_ts[:-1],'target_ts':voxels_ts[:-1, v],'target_ts_deriv':target_derivative[:,v]}
                df=pd.DataFrame(df) # transform in DataFrame for pingouiun toolbox
                seed_to_voxel_correlations[v]=pg.partial_corr(data=df, x='seed_ts', y='target_ts', y_covar='target_ts_deriv').r[0] # compute partial correlation and extract the r 
                
       

        # calculate fisher transformation
        if Fisher == True:
            seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)
                        
                    
          
        return  seed_to_voxel_correlations_fisher_z if Fisher==True else seed_to_voxel_correlations
    

    def mutual_info_maps(self,seed_ts,voxels_ts,output_img=None,save_maps=True,smoothing_output=False,redo=False, n_jobs=1):
        '''
        Create  mutual information maps
        seed_ts: list
            timecourse to use as seed (see extract_data method) (list containing one array per suject)
        target_ts: list
            timecourses of all voxels on which to compute the correlation (see extract_data method) (list containing one array per suject)
        output_img: str
            path + rootname of the output image (/!\ no extension needed) (ex: '/pathtofile/output')
        save_maps: boolean
            to save correlation maps (default = True).
        redo: boolean
            to rerun the analysis
        njobs: int
            number of jobs for parallelization
   
        ----------
        '''
        
        seed_to_voxel_mi=Parallel(n_jobs=n_jobs)(delayed(self._compute_mutual_info)(voxels_ts[subject_nb],seed_ts[subject_nb])
                                           for subject_nb in range(len(self.subject_names)))
        
        print(voxels_ts[0].shape)
        print(seed_ts[0].shape)
        if save_maps==True:
            Parallel(n_jobs=n_jobs)(delayed(self._save_maps)(subject_nb,seed_to_voxel_mi[subject_nb],
                                                                 output_img,
                                                                 smoothing_output)
                                           for subject_nb in range(len(self.subject_names)))
            
            for tmp in glob.glob(os.path.dirname(output_img) + '/tmp_*.nii.gz'):
                os.remove(tmp) # remove temporary 3D images files
                #os.remove(tmp + '.gz') # remove temporary 3D images files
            np.savetxt(os.path.dirname(output_img) + '/subjects_labels.txt',self.subject_names,fmt="%s") # copy the config file that store subject info

           
        
        return seed_to_voxel_mi
    
    def _compute_mutual_info(self,voxels_ts,seed_ts):
        '''
        Run the mutual information analysis:
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#r37d39d7589e2-2
        see also: Cliff et al. 2022 https://arxiv.org/pdf/2201.11941.pdf
        " The Kraskov technique (ksg) combines nearest-neighbor estimators for mutual information based measures"
        
        discrete_features='auto' # will test the sparsity of the data, if the distribution is dense then continuous will be selected
        n_neighbors= 4 # same as Cliff et al. 2022
       
        '''
        
        seed_to_voxel_mi = np.zeros((voxels_ts.shape[1], 1)) # np.zeros(number of voxels,1)
        voxels_ts=np.nan_to_num(voxels_ts,0);seed_ts=np.nan_to_num(seed_ts,0)
        seed_to_voxel_mi = mutual_info_regression(voxels_ts,np.nan_to_num(seed_ts,0),n_neighbors=4)
        #seed_to_voxel_mi=np.nan_to_num(seed_to_voxel_mi,nan=0.0) # replace NaN value by zero
        seed_to_voxel_mi /= np.nanmax(seed_to_voxel_mi) # normalize to the max intensity
        #seed_to_voxel_mi_z=scipy.stats.zscore(seed_to_voxel_mi)# zscored the MI
        
        
        return seed_to_voxel_mi
    
    def distance_corr_maps(self,seed_ts,voxels_ts,output_img=None,save_maps=True,smoothing_output=None,redo=False,n_jobs=1):
        '''
        Compute correlation maps between a seed timecourse and voxels
        Inputs
        ----------
        seed_ts: list
            timecourse to use as seed (see extract_data method) (list containing one array per suject)
        target_ts: list
            timecourses of all voxels on which to compute the correlation (see extract_data method) (list containing one array per suject)
        output_img: str
            path + rootname of the output image (/!\ no extension needed) (ex: '/pathtofile/output')
        Fisher: boolean
            to Fisher-transform the correlation (default = True).
        partial:
            Run partial correlation i.e remove the first derivative on the target signal (default = False).
        save_maps: boolean
            to save correlation maps (default = True).
        redo: boolean
            to rerun the analysis
        njobs: int
            number of jobs for parallelization
    
        Output
        ----------
        output_img_zcorr.nii (if Fisher = True) or output_img_corr.nii (if Fisher = False) 
            4D image containing the correlation maps for all subjects
        subject_labels.txt
            text file containing labels of the subjects included in the correlation analysis
        correlations: array
            correlation maps as an array
        '''

        if not os.path.exists(output_img) or redo==True:
            if not os.path.exists(output_img) or redo==True:
                distance_corr={}
                for subject_nb in range(len(self.subject_names)):
                    distance_corr[subject_nb]=self._compute_distance_corr(seed_ts[subject_nb],voxels_ts[subject_nb],output_img,save_maps)
            
                                                                                             
                    
            if save_maps==True:
                Parallel(n_jobs=n_jobs)(delayed(self._save_maps)(subject_nb,
                                                                 distance_corr[subject_nb],
                                                                 output_img,
                                                                 smoothing_output)
                                           for subject_nb in range(len(self.subject_names)))

            
            
            #for tmp in glob.glob(os.path.dirname(output_img) + '/tmp_*.nii'):
             #   os.remove(tmp) # remove temporary 3D images files

            np.savetxt(os.path.dirname(output_img) + '/subjects_labels.txt',self.subject_names,fmt="%s") # copy the config file that store subject info

        
        #return correlations
        else:
            print("The correlation maps were alredy computed please put redo=True to rerun the analysis")
    
    
    def _compute_distance_corr(self,seed_ts,voxels_ts,output_img,save_maps):
        '''
        Run the distance correlation analyse.
       ----------
       '''
        seed_to_voxel_distCorr = np.zeros((voxels_ts.shape[1], 1))
        
        
        for v in tqdm(range(0,voxels_ts.shape[1]), desc ="process duration"):
                seed_to_voxel_distCorr[v] = dcor.distance_correlation(seed_ts,voxels_ts[:, v])
                
                #voxels_ts.shape[1]
                  
        return seed_to_voxel_distCorr
    
    def _save_maps(self,subject_nb,maps_array,output_img,smoothing):
        '''
        Save maps in a 4D image (one for each participant)
        '''
        
        masker= NiftiMasker(self.mask_target).fit()
        seed_to_voxel_img = masker.inverse_transform(maps_array.T)
        if smoothing is not None:
            seed_to_voxel_img=image.smooth_img(seed_to_voxel_img, smoothing)
            
        seed_to_voxel_img.to_filename(os.path.dirname(output_img) + '/tmp_' + str(subject_nb).zfill(3) +'.nii.gz') # create temporary 3D files
        string='fslmaths ' + os.path.dirname(output_img) + '/tmp_' + str(subject_nb).zfill(3) +'.nii.gz' + ' -mas ' + self.mask_target + ' ' + os.path.dirname(output_img) + '/tmp_' + str(subject_nb).zfill(3) +'.nii.gz'
        os.system(string)
        
        image.concat_imgs(sorted(glob.glob(os.path.dirname(output_img) + '/tmp_*.nii.gz'))).to_filename(output_img + '.nii')
        
        
