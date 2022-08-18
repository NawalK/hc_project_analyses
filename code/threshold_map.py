
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMasker

def Threshold_map(img_filename=None,mask=None,percentile=90):
    '''
    The threshold_map class is used to threshold map of z scored based on a % of voxels
    ex: percentile=90  you'd be trying to identify the lowest score that is greater than 90% of the voxels
    
    Attributes
    ----------
    img_filename : str, default=None
        filename of the input image including the directory (3D or 4D image)
    mask: str, default= None
        None: no masking will be apply, 
        str: filename of the mask including the directory
    percentile: int, default= 0.90
        value between 0 and 1
        it correspond to the % of voxels of interest
        
    Return
    ----------
    threshold: int
        return the value of the threshold to keep n% of the max intense voxels
    nb_voxels: int
        return to the total number of voxels kept
    
        '''
    
    if img_filename is None:
        print("Warning an 3D or 4D image filename should be provide")
        
    if mask is not None:
        masker= NiftiMasker(mask) # extract data from a mask
        data=masker.fit_transform(img_filename)
    else:
        data=nib.load(img_filename).get_fdata()
        print("Warning: if no masking is applied, 0 values will be considered for the thresholding")
        
    data_1D=np.ndarray.flatten(data) # transform the data in an 1D arrat
    threshold=np.percentile(data_1D,percentile)# lowest score that is greater than the percentil        
         
    return threshold
