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




def sort_maps(data, sorting_method,threshold=None):
    ''' Sort maps based on sorting_method (e.g., rostrocaudally)
    
    Inputs
    ----------
    data : array
        4D array containing the k maps to order  
    sorting_method : str
        Method used to sort maps (e.g., 'rostrocaudal', 'rostrocaudal_CoM', 'no_sorting')
    threshold: str
        put a threshold if you're using the 'rostrocaudal_CoM' method
    Output
    ----------
    sort_index : list
        Contains the indices of the sorted maps   
    '''  
    if sorting_method == 'rostrocaudal':
        print('Sorting method: rostrocaudal (max value)')
        max_z = []; 
        for i in range(0,data.shape[3]):
            max_z.append(int(np.where(data == np.nanmax(data[:,:,:,i]))[2][0]))  # take the first max in z direction      
        sort_index = np.argsort(max_z)
        sort_index= sort_index[::-1] # Invert direction to go from up to low
    elif sorting_method == 'rostrocaudal_CoM':
        print('Sorting method: rostrocaudal (center-of-mass biggest cluster)')
        cm_z=[]
        data_thresh =  np.where(data > threshold, data, 0) # Threshold data
        
              
        # We calculate the center of mass of the largest clusters
        for i in range(0,data.shape[3]):
            # Label data to find the different clusters
            lbl1 = label(data_thresh[:,:,:,i])[0]
            cm = center_of_mass(data_thresh[:,:,:,i],lbl1,Counter(lbl1.ravel()).most_common()[1][0]) # Take the center of mass of the larger cluster
            cm_z.append(cm[2])
        
        sort_index = np.argsort(cm_z)
        sort_index= sort_index[::-1] # Invert direction to go from up to low
              
    elif sorting_method == 'no_sorting':
        print('Sorting method: no_sorting')
        sort_index = list(range(data.shape[3]))
    else:
        raise(Exception(f'{sorting_method} is not a supported sorting method.'))
    return sort_index

def match_levels(config, data, method="CoM"):
    ''' Match maps to corresponding spinal levels
    Inputs
    ----------
    config : dict
        Content of the configuration file
    data : array
        4D array containing the k maps to match

    Output
    ----------
    spinal_levels : list
        Array containing one value per map
            C1 = 1, C2 = 2, C3 = 3, C4 = 4, etc.
    '''
    # Find list of spinal levels to consider (defined in config)
        
    levels_list = sorted(glob.glob(config['main_dir'] + config['templates']["sc_levels_path"] + 'spinal_level_*.nii.gz')) # Sorted is used to make sure files are listed f # Sorted is used to make sure files are listed from low to high number (i.e., rostro-caudally)
        
    # Prepare structures
    levels_data = np.zeros((data.shape[0],data.shape[1],data.shape[2],len(levels_list))) # To store spinal levels, based on size of 4D data (corresponding to template) & number of spinal levels in template
    spinal_levels = np.zeros(data.shape[3],dtype='int') # To store corresponding spinal levels

    # Loop through levels & store data
    for lvl in range(0,len(levels_list)):
        level_img = nib.load(levels_list[lvl])
        levels_data[:,:,:,lvl] = level_img.get_fdata()
 
    if method=="CoM":
        map_masked = np.where(data > 1.5, data, 0) # IMPORTANT NOTE: here, a low threshold at 1.5 is used, as the goal is to have rough maps to match to levels
        CoM = np.zeros(map_masked.shape[3],dtype='int')
        for i in range(0,data.shape[3]):
            _,_,CoM[i]=center_of_mass(map_masked[:,:,:,i])
            # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
            level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,CoM[i],:]
            
            spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)
            
    elif method=="max intensity":
        # For each map, find rostrocaudal position of point with maximum intensity
        max_intensity = np.zeros(data.shape[3],dtype='int')
        
        for i in range(0,data.shape[3]):
            max_size=np.where(data == np.nanmax(data[:,:,:,i]))[2].size
            if max_size>1:
                max_intensity[i] = np.where(data == np.nanmax(data[:,:,:,i]))[2][int(max_size/2)] # take the middle max if there are mainy
            else:
                max_intensity[i] = np.where(data == np.nanmax(data[:,:,:,i]))[2]
            
            #print(max_intensity)
            # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
            level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,max_intensity[i],:]
            spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)
    
            
           
        
    else:
        raise(Exception(f'{method} is not a supported matching method.'))
 
    return spinal_levels


          
def tSNR(config,input_files,dataset=None, mask_files=None,outputdir=None,redo=False):
    '''
        Temporal signal/noise ratio calculation; tSNR=mean/std (in time domaine)

        Attributes
        ----------
        config: list
            config file
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
        subject_name=config["list_subjects"][dataset][file_nb]
        if outputdir is not None:
            tSNRdir=outputdir[file_nb]
        else:
            tSNRdir=os.path.dirname(input_files[file_nb])
        
        if dataset=="gva":
            tSNR_file=tSNRdir+ "sub-"+ subject_name+ "_" + os.path.basename(input_files[file_nb]).split('.')[0] + '_tSNR.nii.gz'
        else:
            tSNR_file=tSNRdir+os.path.basename(input_files[file_nb]).split('.')[0] + '_tSNR.nii.gz'
           
        if not os.path.exists(tSNR_file) or redo==True:
            tSNR=image.math_img('img.mean(axis=3) / img.std(axis=3)', img=input_files[file_nb])
            tSNR.to_filename(tSNR_file)
        
        # extract value in a mask
        if mask_files is not None:# and not os.path.exists(tSNR_file.split(".")[0] + "_mean.txt"):
            masker= NiftiMasker(mask_img=mask_files[file_nb],smoothing_fwhm=None) # select the mask
            func_tSNR_masked=masker.fit_transform(tSNR_file) # mask the image
            mean_func_tSNR_masked=np.mean(func_tSNR_masked) # calculate the mean value
            
            with open(tSNR_file.split(".")[0] + "_mean.txt", 'w') as f:
                f.write(str(mean_func_tSNR_masked))  # save in a file

            #extract
            
                         
        tSNR_files.append(tSNR_file)
        tSNR_means.append(mean_func_tSNR_masked)
    return tSNR_files, tSNR_means

def unzip_file(self,input_files,ext=".nii",zip_file=False, redo=False):
        '''
        unzip the file to match with SPM
        Attributes
        ----------
        input_files: list
            list of input files (one for each participants)
        ext: extension after unzip
            default: ".nii", put ".nii.gz" to zip a file
        zip_file: Bolean
            zip the file instead of unzip a file
        redo: Bolean
                optional, to rerun the analysis put True (default: False)
        return
        ----------
        output_files: list
            list of unziped or zipped files (one for each participants)
        '''
        output_files=[]
        for file_nb in range(0,len(input_files)):
            if zip_file== False:
                if not os.path.exists(input_files[file_nb].split('.')[0] + ext) or redo==True:
                    input = gzip.GzipFile(input_files[file_nb], 'rb') # load the  .nii.gz
                    s = input.read(); input.close()
                    unzip = open(input_files[file_nb].split('.')[0] + ext, 'wb') # save the .nii
                    unzip.write(s); unzip.close()
                    print('Unzip done for: ' + os.path.basename(input_files[file_nb]))
                else:
                    print("Unzip was already done please put redo=True to redo that step")

                output_files.append(input_files[file_nb].split('.')[0] + ext)
            
            elif zip_file== True:
                if not os.path.exists(input_files[file_nb].split('.')[0] + ext) or redo==True:
                    string= 'gzip ' + input_files[file_nb]
                    os.environ(string)
                #else:
                    #print("Zip was already done please put redo=True to redo that step")

                output_files.append(input_files[file_nb].split('.')[0] + ext)
         
               
  
        
        
        return output_files
    
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
