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


import sys,os
import json
# Spinal cord Toolbox_________________________________________
### Cerebro:
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.0.0")
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.0.0/scripts") #sys.path.insert(0, "/cerebro/cerebro1/dataset/bmpd/derivatives/sc_preproc/code/sct/spinalcordtoolbox")
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/code/") #sys.path.insert(0, "/cerebro/cerebro1/dataset/bmpd/derivatives/sc_preproc/code/sct/spinalcordtoolbox")

from spinalcordtoolbox.utils.sys import run_proc

from canICA_analyses import ICA



# ## <font color=#00988c>  Run the ICA analysis </font>

# In[11]:

os.chdir("/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/code/") #sys.path.insert(0, "/cerebro/cerebro1/dataset/bmpd/derivatives/sc_preproc
# Load the dataset config
#config_BMPD_HC_CL.json
with open('../config/config_SPiCiCAP_CL.json') as config_file:
    config = json.load(config_file)
    
for k in config["k_range"]["spinalcord"]:
    config["n_comp"]=k # usefull if you want to test only on k
    print(config["n_comp"])
    icas = ICA("spinalcord",config) # "brain_spinalcord" or "brain" or "spinalcord"
    #all_data=icas.get_data(run='extract',t_r=config["t_r"],n_jobs=8) # load or extract
    all_data=icas.get_data(run='load',t_r=config["t_r"],n_jobs=8) # load or extract
    reducedata_all=icas.indiv_PCA(all_data,save_indiv_img=True)
    components=icas.get_CCA(reducedata_all)
    components_final,components_final_z=icas.get_ICA(components)
    zcomponents4D_filename=icas.save_components(components_final,components_final_z)
    