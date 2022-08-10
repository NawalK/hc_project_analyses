import glob, os, shutil
import nibabel as nib
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn import image
from joblib import Parallel, delayed
from sklearn import decomposition


# to improve---------------------------
# loop per seed
# brain > sc or sc to brain

# extract fct: 
# plotting
#

class Seed2voxels:
    '''
    The Seed2voxels class is used to run correlation analysis
    Attributes
    ----------
    config : dict
    seed_names: 2D array
        name of the seeds (ex: ['spinalcord_seed1','spinalcord_seed2'])
    target_name: 2D array
        name of the target structure (ex: ['brain_GM'])
    
        '''
    
    def __init__(self, config, seed_names, target_name):
        self.config = config # load config info
        self.subject_names= config ["list_subjects"]
        self.outputdir= self.config["main_dir"] +self.config["seed2vox_dir"]
        self.seed_names=seed_names
        self.target_name=target_name
    
   
        
    def extract_data(self,img=None,mask=None,smoothing_fwhm=None, timeseries_txt=None,run="extract",n_jobs=1):
    '''
    Extracts or load time series in a mask 
    Attributes
    ----------
    img: list
        list of 4d image filenames (ex: ['sbj1.nii.gz','sbj2.nii.gz'])
    mask: str
        filename of the mask uses to extract the timeseries (ex: 'mask.nii.gz')
    smoothing_fwhm: array
        to apply smoothing during the extraction (ex: [6,6,6])
    timeseries_txt: list
        list of the output filenames, should be the same lentght as the img (ex: ['ts_sbj1.txt','ts_sbj2.txt'])
    run: "extract" or "load"
        to run timeserie extraction or load a file (timeseries_txt) that contain the timeseries
   
    Return
    ----------
    timeseries:
        timeseries of each voxels for each participant
    timeseries_mean:
        mean timeserie in the mask for each participant
    timeseries_pc1:
        principal component in the mask for each participant
    '''
        timeseries_mean=[]; timeseries=[];timeseries_pc1=[]
        self.img=img
        self.mask=mask
        self.smoothing_fwhm=smoothing_fwhm
        self.timeseries_txt=timeseries_txt
        self.job=n_jobs
        
        if run=="extract":
            ## Extract data in the seed___________________________________________
            ts=Parallel(n_jobs=n_jobs)(delayed(self._extract_ts)(subject_nb)
                                       for subject_nb in range(len(self.subject_names)))
            
            for subject_nb in range(len(self.subject_names)):
                timeseries.append(ts[subject_nb][0])
                timeseries_mean.append(ts[subject_nb][1])
                timeseries_pc1.append(ts[subject_nb][2])
                 
        elif run=="load":
            timeseries=Parallel(n_jobs=n_jobs)(delayed(np.loadtxt)(timeseries_txt[subject_nb] + '.txt') for subject_nb in range(len(self.subject_names)))
            timeseries_mean=Parallel(n_jobs=n_jobs)(delayed(np.loadtxt)(timeseries_txt[subject_nb] + '_mean.txt') for subject_nb in range(len(self.subject_names)))
            timeseries_pc1=Parallel(n_jobs=n_jobs)(delayed(np.loadtxt)(timeseries_txt[subject_nb] + '_PC1.txt') for subject_nb in range(len(self.subject_names)))
           
        else:
            print("## Use run='extract' or run='load'")
        
        return timeseries,timeseries_mean,timeseries_pc1


        
    def _extract_ts(self,subject_nb):
    '''
    Extracts time series in a mask + calculates the mean
    Attributes
    ----------
    img: list
        list of 4d image filenames (ex: ['sbj1.nii.gz','sbj2.nii.gz'])
    mask: str
        filename of the mask uses to extract the timeseries (ex: 'mask.nii.gz')
    smoothing_fwhm: array
        to apply smoothing during the extraction (ex: [6,6,6])
    timeseries_txt: list
        list of the output filenames, should be the same lentght as the img (ex: ['ts_sbj1.txt','ts_sbj2.txt'])
    run: "extract" or "load"
        to run timeserie extraction or load a file (timeseries_txt) that contain the timeseries
   
    Return
    ----------
    timeseries:
        timeseries of each voxels for each participant
    timeseries_mean:
        mean timeserie in the mask for each participant
    timeseries_pc1:
        principal component in the mask for each participant
    '''
        masker= NiftiMasker(self.mask,smoothing_fwhm=self.smoothing_fwhm) # seed masker
        ts=masker.fit_transform(self.img[subject_nb])
        ts_mean=np.mean(ts,axis=1) # mean time serie
        
        #calculate the principal component in the seed
        pca=decomposition.PCA(n_components=1)
        pca_components=pca.fit_transform(ts)
        ts_pc1=pca_components[:,0]
       
        np.savetxt(self.timeseries_txt[subject_nb] + '.txt',ts)
        np.savetxt(self.timeseries_txt[subject_nb] + '_mean.txt',ts_mean)
        np.savetxt(self.timeseries_txt[subject_nb] + '_PC1.txt',ts_pc1)
        
        print('Timeseries extraction for subject: ' + self.subject_names[subject_nb] + ' done')
        
        return ts, ts_mean, ts_pc1

        
    def correlation_maps(self,seed_ts_mean,voxels_ts,mask,output_img,Fisher=True,n_jobs=1):
        self.seed_ts_mean=seed_ts_mean;self.voxels_ts=voxels_ts
        self.mask=mask; self.output_img=output_img; self.Fisher=Fisher
        self.job=n_jobs
        Parallel(n_jobs=n_jobs)(delayed(self._compute_correlation)(subject_nb)
                                       for subject_nb in range(len(self.subject_names)))
                               
       
        # transform of all participant in a 4D image
        image.concat_imgs(glob.glob(os.path.dirname(output_img) + '/tmp_*.nii')).to_filename(output_img)
        for tmp in glob.glob(os.path.dirname(output_img) + '/tmp_*.nii'):
            os.remove(tmp) # remove temporary 3D images files
        
        np.savetxt(os.path.dirname(output_img) + 'subjects_labels.txt',self.subject_names,fmt="%s") # copy the config file that store subject info

    def _compute_correlation(self,subject_nb):
        seed_to_voxel_correlations = np.zeros((self.voxels_ts[subject_nb].shape[1], 1)) # np.zeros(number of voxels,1)
        for v in range(0,self.voxels_ts[subject_nb].shape[1]): 
            # compute correlation
            seed_to_voxel_correlations[v] = np.corrcoef(self.seed_ts_mean[subject_nb], self.voxels_ts[subject_nb][:, v])[0, 1]
            
        # calculate fisher transformation
        if self.Fisher == True:
            seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)
            masker= NiftiMasker(self.mask).fit()
            seed_to_voxel_correlations_fisher_z_img = masker.inverse_transform(seed_to_voxel_correlations_fisher_z.T)
            seed_to_voxel_correlations_fisher_z_img.to_filename(os.path.dirname(self.output_img) + '/tmp_' + str(subject_nb) +'.nii') # create temporary 3D files
