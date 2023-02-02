#!/usr/bin/env python
# coding: utf-8

# # <font color=#f4665a>  ICA analysis </font>

# ### Project: BMPD HC
# ____________________________________________________
# 
# **Description:** This notebook provides code for BOLD signal fMRI resting-state processing for the Biomarker for Parkinson's Disease (BMPD)data. 
# We will used ICA analysis:
# CanICA is an ICA package for group-level analysis of fMRI data.  
# It brings a well-controlled group model, as well as a thresholding algorithm controlling for specificity and sensitivity with an explicit model of the signal.  
# The reference papers are: G. Varoquaux et al. "A group model for stable multi-subject ICA on fMRI datasets", NeuroImage Vol 51 (2010), p. 288-299
# G. Varoquaux et al. "ICA-based sparse features recovery from fMRI datasets", IEEE ISBI 2010, p. 1177
# 
# 
# **Toolbox required:** SpinalCordToolbox, FSL, nilearn toolbox, nipype, matlab
# 
# **Inputs**:  
# This notebook required this the following prepross anatomical,fmri images 
# 
# **Ouputs**:
# See the output description at each step of the Notebook.
# 
# ____________________________________________________
# 

# ## <font color=#00988c> Imports </font>

# In[1]:


import sys, os
import json
# Spinal cord Toolbox_________________________________________
### Cerebro:
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.0.0")
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.0.0/scripts") #sys.path.insert(0, "/cerebro/cerebro1/dataset/bmpd/derivatives/sc_preproc/code/sct/spinalcordtoolbox")
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/code/") #sys.path.insert(0, "/cerebro/cerebro1/dataset/bmpd/derivatives/sc_preproc/code/sct/spinalcordtoolbox")

from spinalcordtoolbox.utils.sys import run_proc

from nilearn.maskers import NiftiMasker


from canICA_analyses import ICA


# ## <font color=#00988c>  I. Run the canICA analysis </font>

# In[19]:
os.chdir("/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/code/")

# Load the dataset config
#config_spine_only_CL.json #../config/config_brsc_CL.json
with open('../config/config_spine_only_CL.json') as config_file:
    config = json.load(config_file)
dataset="gva" 
structures=["spinalcord"] # ["spinalcord"] or ["brain","spinalcord"] . double check the script for brainsc
config["ica_ana"]["iter"]


# In[3]:


import glob
files_func={};func_allsbj={}
for structure in structures:
    if len(structures) == 1:
        ana=structure
    else:
        ana= "brain_spinalcord"
    files_func[structure]=[];func_allsbj[structure]=[]
    for sbj_nb in range(len(config["list_subjects"][dataset])):
        subject_name=config["list_subjects"][dataset][sbj_nb]
        files_func[structure].append(glob.glob(config["data"][dataset]["inputs_ica"]["dir"]+ '/sub-' + subject_name + '/'  + structure + '/*' + config["data"][dataset]["inputs_ica"][structure]["tag_filename_" + ana] + '*')[0])
       


outputdir=[]
config["ica_ana"]["iter"]=500
for sbj_nb in range(0,len(config["list_subjects"][dataset])):
    subject_name=config["list_subjects"][dataset][sbj_nb]
    print("fast ICA is running for " + subject_name)
    config["list_subjects"][dataset]=[subject_name]
    config["ica_ana"]["n_comp"]=5
    
        
    icas = ICA([files_func[structure][sbj_nb]],[''],structures,dataset,config) 
    
    # extract individual data
    components=icas.get_data(run='load',t_r=config["acq_params"][dataset]["TR"],n_jobs=8) # load or extract, if NaN issues put both
    components_final,components_final_z=icas.get_ICA(components[0].T,k=9) # components (n_voxels,n_volumes)
    zcomponents4D_filename=icas.save_components(components_final,components_final_z,one_subject=subject_name)

