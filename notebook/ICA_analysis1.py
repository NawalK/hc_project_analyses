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
os.chdir("/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/code/")
from canICA_analyses import ICA

# ## <font color=#00988c>  Run the ICA analysis </font>

# In[2]:


# Load the dataset config
#config_spine_only_CL.json #../config/config_brsc_CL.json
with open('../config/config_spine_only_CL.json') as config_file:
    config = json.load(config_file)
dataset="gva" 
structures=["spinalcord"] # ["spinalcord"] or ["brain","spinalcord"] . double check the script for brainsc


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
        


# In[35]:


for k in config["ica_ana"]["k_range"]["spinalcord"]:
    config["ica_ana"]["n_comp"]=k # usefull if you want to test only on k
    print(config["ica_ana"]["n_comp"])
    icas = ICA(files_func[structure],[''],structures,dataset,config) # "brain_spinalcord" or "brain" or "spinalcord"
    #icas = ICA(files_func[structures[0]],files_func[structures[1]],structures,dataset,config) # "brain_spinalcord" or "brain" or "spinalcord"
    if k==4:
        all_data=icas.get_data(run='extract',t_r=config["acq_params"][dataset]["TR"],n_jobs=1) # load or extract
    all_data=icas.get_data(run='load',t_r=config["acq_params"][dataset]["TR"],n_jobs=8) # load or extract
    reducedata_all=icas.indiv_PCA(all_data,save_indiv_img=True) # that step is not implanted to save individual maps for brain + sc yet
    components=icas.get_CCA(reducedata_all)
    components_final,components_final_z=icas.get_ICA(components)
    zcomponents4D_filename=icas.save_components(components_final,components_final_z)
    
    

