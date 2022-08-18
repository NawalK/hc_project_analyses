import glob, os, shutil
from urllib.parse import non_hierarchical
import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker
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
    signal: str
        type of signal ('raw' for bold or 'ai' for deconvoluted)
    seed_names: 2D array
        name of the seeds (ex: ['spinalcord_seed1','spinalcord_seed2'])
    target_name: 2D array
        name of the target structure (ex: ['brain_GM'])
    '''
    
    def __init__(self, config, signal, seed_names, target_name):
        self.config = config # load config info
        self.signal=signal
        self.subject_names= config ["list_subjects"]
        self.outputdir= self.config["main_dir"] +self.config["seed2vox_dir"]
        self.seed_names=seed_names
        self.target_name=target_name
    
    def extract_data(self,img=None, mask=None,smoothing_fwhm=None,timeseries_txt=None,run="extract",n_jobs=1):
        '''
        Extracts time series in a mask or load them from text files
        Inputs
        ----------
        img: list
            list of 4d image filenames on which to extract signals (ex: ['sbj1.nii.gz','sbj2.nii.gz'])
        mask: str
            filename of the mask uses to extract the timeseries (ex: 'mask.nii.gz')
        smoothing_fwhm: array
            to apply smoothing during the extraction (ex: [6,6,6])
        timeseries_txt: list
            if run == "load" 
                list of the output filenames, should be the same length as the img (ex: ['ts_sbj1.txt','ts_sbj2.txt'])
            if run == "extract" 
                list of the filenames to load (ex: ['ts_sbj1.txt','ts_sbj2.txt'])
        run: "extract", "load"
            to run timeserie extraction or load a file (timeseries_txt) that contain the timeseries
            Note: to load iCAP-based timecourses, use "load_ai", as mean signal and pc1 are not
    
        Returns
        ----------
        timeseries:
            timeseries of each voxels for each participant
        timeseries_mean:
            mean timeserie in the mask for each participant
        timeseries_pc1:
            principal component in the mask for each participant
        '''
        timeseries_mean=[]; timeseries=[]; timeseries_pc1=[]

        # Signals are extracted differently for ica and icaps
        if self.signal == 'raw':
            if run == "extract":
                ## Extract data in the seed___________________________________________
                ts=Parallel(n_jobs=n_jobs)(delayed(self._extract_ts)(mask,img[subject_nb],timeseries_txt[subject_nb],smoothing_fwhm)
                                        for subject_nb in range(len(self.subject_names)))
                
                for subject_nb in range(len(self.subject_names)):
                    timeseries.append(ts[subject_nb][0])
                    timeseries_mean.append(ts[subject_nb][1])
                    timeseries_pc1.append(ts[subject_nb][2])
                    
            elif run == "load":
                for subject_nb in range(len(self.subject_names)):
                    timeseries=Parallel(n_jobs=n_jobs)(delayed(np.loadtxt)(timeseries_txt[subject_nb] + '.txt') for subject_nb in range(len(self.subject_names)))
                    timeseries_mean=Parallel(n_jobs=n_jobs)(delayed(np.loadtxt)(timeseries_txt[subject_nb] + '_mean.txt') for subject_nb in range(len(self.subject_names)))
                    timeseries_pc1=Parallel(n_jobs=n_jobs)(delayed(np.loadtxt)(timeseries_txt[subject_nb] + '_PC1.txt') for subject_nb in range(len(self.subject_names)))
  
            else:
                raise(Exception(f"Use run='extract' or run='load'"))

        elif self.signal == 'ai':
            if run == "load":
                timeseries=Parallel(n_jobs=n_jobs)(delayed(np.loadtxt)(timeseries_txt[subject_nb] + '.txt') for subject_nb in range(len(self.subject_names)))
            
            elif run == "extract":
                print('Note: signals can only be extracted for targets for activity-inducing signals.')
                timeseries = Parallel(n_jobs=n_jobs)(delayed(self._extract_ts)(mask,img[subject_nb],timeseries_txt[subject_nb],smoothing_fwhm)
                                        for subject_nb in range(len(self.subject_names)))
                
            else:
                raise(Exception(f"Use run='extract' or run='load'"))
        else:
            raise(Exception(f"Use signal='raw' or signal='ai'"))
        
        # For AI-based (i.e. for iCAP pipeline), no need to return timeseries mean and pc1
        return timeseries if self.signal=='ai' else (timeseries,timeseries_mean,timeseries_pc1)

    def _extract_ts(self,mask,img,ts_txt,smoothing=None):
        '''
        Extracts time series in a mask + calculates the mean and PC
        '''
        masker= NiftiMasker(mask,smoothing_fwhm=smoothing, t_r=1.55,low_pass=0.17 if self.signal == 'raw' else None, high_pass=0.01 if self.signal == 'raw' else None) # seed masker
        ts=masker.fit_transform(img)
        np.savetxt(ts_txt + '.txt',ts)

        # For raw signal, compute mean and PC
        if self.signal=='raw':
            ts_mean=np.mean(ts,axis=1) # mean time serie
            pca=decomposition.PCA(n_components=1)
            pca_components=pca.fit_transform(ts)
            ts_pc1=pca_components[:,0]
            np.savetxt(ts_txt + '_mean.txt',ts_mean)
            np.savetxt(ts_txt + '_PC1.txt',ts_pc1)
                
        return ts if self.signal=='ai' else (ts, ts_mean, ts_pc1)

    def correlation_maps(self,seed_ts,voxels_ts,output_img,mask,Fisher=True,n_jobs=1):
        '''
        Compute correlation maps between a seed timecourse and a voxelwise one
        Inputs
        ----------
        seed_ts: list
            timecourse to use as seed (see extract_data method) (list containing one array per suject)
        target_ts: list
            timecourses of all voxels on which to compute the correlation (see extract_data method) (list containing one array per suject)
        mask: str
            path of the mask uses to extract the timeseries (ex: '/pathtofile/mask.nii.gz')
        output_img: str
            path + rootname of the output image (/!\ no extension needed) (ex: '/pathtofile/output')
        Fisher: boolean
            to Fisher-transform the correlation (default = True).
        njobs: int
            number of jobs for parallelization
    
        Output
        ----------
        output_img_zcorr.nii (if Fisher = True) or output_img_corr.nii (if Fisher = False) 
            4D image containing the correlation maps for all subjects
        subject_labels.txt
            text file containing labels of the subjects included in the correlation analysis
        '''
        Parallel(n_jobs=n_jobs)(delayed(self._compute_correlation)(subject_nb,voxels_ts[subject_nb],seed_ts[subject_nb],mask,output_img,Fisher)
                                       for subject_nb in range(len(self.subject_names)))
                               
        # transform of all participant in a 4D image
        if Fisher == True:
            image.concat_imgs(glob.glob(os.path.dirname(output_img) + '/tmp_*.nii')).to_filename(output_img + '_zcorr.nii')
        elif Fisher == False:
            image.concat_imgs(glob.glob(os.path.dirname(output_img) + '/tmp_*.nii')).to_filename(output_img + '_corr.nii')
        else:
            raise(Exception(f"Fisher should be True or False"))

        for tmp in glob.glob(os.path.dirname(output_img) + '/tmp_*.nii'):
            os.remove(tmp) # remove temporary 3D images files
        
        np.savetxt(os.path.dirname(output_img) + '/subjects_labels.txt',self.subject_names,fmt="%s") # copy the config file that store subject info

    def _compute_correlation(self,subject_nb,voxels_ts,seed_ts,mask,output_img,Fisher):
        seed_to_voxel_correlations = np.zeros((voxels_ts.shape[1], 1)) # np.zeros(number of voxels,1)
        for v in range(0,voxels_ts.shape[1]): 
            # compute correlation
            seed_to_voxel_correlations[v] = np.corrcoef(seed_ts, voxels_ts[:, v])[0, 1]
        
        # prepare mask for saving
        masker= NiftiMasker(mask).fit()

        # calculate fisher transformation
        if Fisher == True:
            seed_to_voxel_correlations_fisher_z = np.arctanh(seed_to_voxel_correlations)
            seed_to_voxel_correlations_fisher_z_img = masker.inverse_transform(seed_to_voxel_correlations_fisher_z.T)
            seed_to_voxel_correlations_fisher_z_img.to_filename(os.path.dirname(output_img) + '/tmp_' + str(subject_nb) +'.nii') # create temporary 3D files
        elif Fisher == False:
            seed_to_voxel_correlations_fisher_img = masker.inverse_transform(seed_to_voxel_correlations.T)
            seed_to_voxel_correlations_fisher_img.to_filename(os.path.dirname(output_img) + '/tmp_' + str(subject_nb) +'.nii') # create temporary 3D files
        else:
            raise(Exception(f"Fisher should be True or False"))