#!/usr/bin/env python
# coding: utf-8

# ## <font color=#B2D732> <span style="background-color: #4424D6"> Imports

# In[80]:


import sys,json
import glob, os
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/code/")

sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.0.0")
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.0.0/scripts") 
from spinalcordtoolbox.utils.sys import run_proc

from seed_to_voxels import Seed2voxels


# ## <font color=#B2D732> <span style="background-color: #4424D6"> A/ Initialization

#  ### <font color=#4424D6> I. Configuration & parameters </font >

# In[81]:


# Load config file ------------------------------------------------------------
with open('/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/config/config_seed2voxels.json') as config_file: # the notebook should be in 'xx/notebook/' folder #config_proprio
    config = json.load(config_file) # load config file should be open first and the path inside modified
    #config['list_subjects']=config["list_subjects_younger"]
signal='raw'


# ### <font color=#4424D6> II. Initialize class based on this </font>

# In[82]:


seed2voxels=Seed2voxels(config,signal) # initialize the function


# ## <font color=#B2D732> <span style="background-color: #4424D6"> B/ Data extraction </font></span>
# ### <font color=#4424D6> I. Time series extraction - Target </font>
# 

# In[103]:


target_timeseries,seeds_timeseries =seed2voxels.extract_data(redo=True,n_jobs=8)


output_img={};
for seed_name in config["seeds"]["seed_names"]:
    output_img[seed_name]=[]
    output_img[seed_name]=config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/' + seed_name + '/' + config["targeted_voxels"]["target_name"]+ '_fc_maps/corr_' + str(len(config['list_subjects'])) + 'subjects_seed_' + seed_name + '_s'

    corr=seed2voxels.correlation_maps(seeds_timeseries["zmean"][seed_name],target_timeseries["zscored"],output_img=output_img[seed_name],Fisher=True,partial=False,save_maps=True,smoothing_output=[6,6,6],redo=True,n_jobs=1)
