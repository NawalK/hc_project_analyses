#!/usr/bin/env python
# coding: utf-8

# ## <font color=#B2D732> <span style="background-color: #4424D6"> Imports

# In[1]:
#cd /cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/notebook/brain_spine/
#nohup python /cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/notebook/brain_spine/seed_to_voxels_br_sc_CL.py /cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/notebook/brain_spine/ .pynohup.out &


import sys,json
import glob, os
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/code/")

from seed_to_voxels import Seed2voxels

# In[2]:


## <font color=#B2D732> <span style="background-color: #4424D6"> A/ Initialization


#  ### <font color=#4424D6> I. Configuration & parameters </font >

# In[20]:


# Load config file ------------------------------------------------------------
with open('/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/config/config_seed2voxels.json') as config_file: # the notebook should be in 'xx/notebook/' folder #config_proprio
    config = json.load(config_file) # load config file should be open first and the path inside modified
    #config['list_subjects']=config["list_subjects_younger"]

seed_indiv=False


# ### <font color=#4424D6> II. Initialize class based on this </font>

# In[21]:


seed2voxels=Seed2voxels(config,seed_indiv) # initialize the function


# ## <font color=#B2D732> <span style="background-color: #4424D6"> B/ Data extraction </font></span>
# ### <font color=#4424D6> I. Time series extraction - Target </font>
# 

# In[23]:


target_timeseries,seeds_timeseries =seed2voxels.extract_data(redo=True,n_jobs=8) 


# ## <font color=#B2D732> <span style="background-color: #4424D6"> C/ Correlation analysis

# In[24]:


for side in ["positive","negative","two-sided"]:
    output_dir={};output_file={};

    if side=="positive":
        tag="pos"
    elif side=="negative":
        tag="neg"
    elif side=="two-sided":
        tag="bi"

    for seed_name in config["seeds"]["seed_names"]:
        output_dir[seed_name]=config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/' + seed_name + '/' + config["targeted_voxels"]["target_name"]+ '_fc_maps/Corr/'
        if not os.path.exists(output_dir[seed_name]):
            os.mkdir(output_dir[seed_name])

        output_file[seed_name]=  output_dir[seed_name] +'/'+tag+'-corr_' + str(len(config['list_subjects'])) + 'subjects_seed_' + seed_name 

        # run correlation analysis
        corr=seed2voxels.correlation_maps(seeds_timeseries["zmean"][seed_name],
                                          target_timeseries["zscored"],
                                          output_img=output_file[seed_name],
                                          Fisher=True,
                                          partial=False,
                                          side=side,
                                          save_maps=True,
                                          smoothing_output=None,
                                          redo=True,
                                          n_jobs=8)

        #calculate the mean across participant
        string="fslmaths " +output_file[seed_name] + " -Tmean " + output_file[seed_name].split(".")[0] + "_mean.nii.gz"
        os.system(string)
   

   
