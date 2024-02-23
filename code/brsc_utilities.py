import glob, os, sys, gzip
import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass,label
from collections import Counter 
from scipy import stats
from joblib import Parallel, delayed
# nilearn toolbox
from nilearn import image
from nilearn.maskers import NiftiMasker


          
def tSNR(input_files,dataset=None, mask_files=None,outputdir=None,redo=False):
    '''
        Temporal signal/noise ratio calculation; tSNR=mean/std (in time domaine)

        Attributes
        ----------
        input_files : list
            list of 4D input files (one for each participants)
        dataset: str
            Name of the dataset (default=None)
        mask_files: list
            list of 3D mask files if you want to calculate the tSNR in a mask (one for each participants)
            (default=None)
        out_dir: list
            list of output directories (one for each participants)
             
       return
       ----------
       tSNR_files: list
            list of tSNR files (one for each participants)
       
       '''
    
    tSNR_files=[]; tSNR_means=[]
    for file_nb in range(0,len(input_files)):
        if outputdir is not None:
            tSNRdir=outputdir[file_nb]
        else:
            tSNRdir=os.path.dirname(input_files[file_nb])
        
        tSNR_file=tSNRdir+os.path.basename(input_files[file_nb]).split('.')[0] + '_tSNR.nii.gz'

        if not os.path.exists(tSNR_file) or redo==True:
            print("running")
            tSNR=image.math_img('img.mean(axis=3) / img.std(axis=3)', img=input_files[file_nb])
            tSNR.to_filename(tSNR_file)
        
        # extract value in a mask
        #if mask_files is not None:# and not os.path.exists(tSNR_file.split(".")[0] + "_mean.txt"):
         #   masker= NiftiMasker(mask_img=mask_files[file_nb],smoothing_fwhm=None) # select the mask
          #  func_tSNR_masked=masker.fit_transform(tSNR_file) # mask the image
           # mean_func_tSNR_masked=np.mean(func_tSNR_masked) # calculate the mean value
            
            #with open(tSNR_file.split(".")[0] + "_mean.txt", 'w') as f:
             #   f.write(str(mean_func_tSNR_masked))  # save in a file

            #extract
            
                         
        tSNR_files.append(tSNR_file)
        #tSNR_means.append(mean_func_tSNR_masked)
    return tSNR_files#, tSNR_means

    
class Preproc:
    def __init__(self,config):
        self.config = config # load config info
        # imports

        # Spinal cord Toolbox_________________________________________
        sys.path.append(config["tools_dir"]["sct_toolbox"]); sys.path.append(config["tools_dir"]["sct_toolbox"] +"/scripts")
        sys.path.append('../code/') # should be change
        # spinal cord toolbox
        from spinalcordtoolbox.utils.sys import run_proc
        
    def normalisation_sc(self,warp_files,func_files,func_norm_files,sc_template,n_jobs=1,redo=False):
        '''
            Normalise spinalcord input image into PAM50 a destination image using sct_apply_transfo (sct toolbox)
        Attributes
        ----------
        warping_fields : list
                    list of 4D warping field files (one for each participants)  
        func_files : list
                    list of 4D input files (one for each participants)

        func_norm_files : list
                    list of 4D output files (one for each participants)

        sc_template: list
                template image (T2w or T1w)
        n_job: int
            number of parallele jobs
        redo: Bolean
                optional, to rerun the analysis put True (default: False)

        '''

        if not os.path.exists(func_norm_files[0]) or redo==True:

            Parallel(n_jobs=n_jobs)(delayed(self._run_normalisation_sc)(func_files[sbj_nb],
                                                                             sc_template,
                                                                            warp_files[sbj_nb],
                                                                            func_norm_files[sbj_nb])
                                            for sbj_nb in range(len(warp_files)))

            return print('Normalisation into PAM50 space done')
        else:
            return print('Normalisation into PAM50 space was already done, set redo=True to run it again')

    def _run_normalisation_sc(self,func_file,sc_template,warp_file,func_norm_file):

            run_proc('sct_apply_transfo -i {} -d {} -w {} -x spline -o {}'.format(func_file,
                                      sc_template,
                                      warp_file,
                                      func_norm_file))

            print("Normalisation done: " + os.path.basename(func_norm_file))
