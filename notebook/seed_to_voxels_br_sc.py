#!/usr/bin/env python
# coding: utf-8

# ## <font color=#B2D732> <span style="background-color: #4424D6"> Imports

# In[137]:

import sys,json
import glob, os
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/code/")

from seed_to_voxels import Seed2voxels

# ## <font color=#B2D732> <span style="background-color: #4424D6"> Initialization

#  ### <font color=#4424D6> I. Configuration & parameters </font >

# In[138]:


with open('../config/config_CL.json') as config_file:
    config = json.load(config_file) # load config file
signal='raw'
seed_names=['spinalcord_C1','spinalcord_C2','spinalcord_C3','spinalcord_C4','spinalcord_C5','spinalcord_C6','spinalcord_C7','spinalcord_C8','spinalcord_C9'] # define seed names ex: 'spinalcord_ICA-C4'
seed_names=['spinalcord_C1-RD','spinalcord_C1-LD','spinalcord_C1-RV','spinalcord_C1-LV','spinalcord_C2-RD','spinalcord_C2-LD','spinalcord_C2-RV','spinalcord_C2-LV','spinalcord_C3-RD','spinalcord_C3-LD','spinalcord_C3-RV','spinalcord_C3-LV','spinalcord_C4-RD','spinalcord_C4-LD','spinalcord_C4-RV','spinalcord_C4-LV','spinalcord_C5-RD','spinalcord_C5-LD','spinalcord_C5-RV','spinalcord_C5-LV','spinalcord_C6-RD','spinalcord_C6-LD','spinalcord_C6-RV','spinalcord_C6-LV','spinalcord_C7-RD','spinalcord_C7-LD','spinalcord_C7-RV','spinalcord_C7-LV'] #['brain_BA4']#['spinalcord_C5-whole'] # define seed names ex: 'spinalcord_ICA-C4'

target_name=['brain_mask'] # define structure target


# ### <font color=#4424D6> II. Select files </font>

# In[139]:


# One target per subject
data_target=[];ts_target_txt=[];ts_seed_txt ={};data_seed=[];mask_seed={}

for subject_name in config['list_subjects']:
    data_target.append(glob.glob(config["coreg_dir"] + 'sub-'+ subject_name +'/'+target_name[0].split('_')[0]+'/*' + config["coreg_tag"][target_name[0].split('_')[0]] +'*.gz')[0])
    ts_target_txt.append(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+target_name[0]+'/timeseries/sub_' + subject_name + '_mask_' + target_name[0] + '_timeseries')
    data_seed.append(glob.glob(config["coreg_dir"] + 'sub-'+ subject_name +'/'+seed_names[0].split('_')[0]+'/*' + config["coreg_tag"][seed_names[0].split('_')[0]] +'*')[0])

for seed_name in seed_names:
    ts_seed_txt[seed_name]=[]
    for subject_name in config['list_subjects']:
        ts_seed_txt[seed_name].append(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name+'/timeseries/sub_' + subject_name + '_mask_' + seed_name.split('_')[-1] + '_timeseries')

for seed_name in seed_names:
    mask_seed[seed_name]=glob.glob(config["main_dir"] + config["data"]["ICA"]["spinalcord_dir"]+ 'K_9/comp_rois/' +  '*' + seed_name.split('_')[1] + '*')[0]
#config["main_dir"] + '/templates/MNI/rois/L-BA4.nii.gz'#    
print(mask_seed[seed_name])
mask_target=glob.glob(config["main_dir"] + config["masks"][target_name[0].split('_')[0]])[0]

# create output directory if needed
for seed_name in seed_names:
    if not os.path.exists(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name):
        os.mkdir(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name)
        os.mkdir(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name+'/timeseries/') # folder to store timeseries extraction
        os.mkdir(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name+'/brain_fc_maps/') # folder to store maps of FC


# ### <font color=#4424D6> III. Initialize class based on this </font>

# In[140]:


for seed_name in seed_names:
    seed2voxels=Seed2voxels(config,signal,seed_name,target_name) # initialize the function


# ## <font color=#B2D732> <span style="background-color: #4424D6"> Data extraction </font></span>
# ### <font color=#4424D6> I. Time series extraction - Target </font>
# 

# In[ ]:


target_timeseries,target_timeseries_mean,target_timeseries_pc1=seed2voxels.extract_data(img=data_target, mask=mask_target, timeseries_txt=ts_target_txt,
                                                                run="load",n_jobs=8,smoothing_fwhm=[6,6,6]) # run the analyse for target voxels


# ### <font color=#4424D6> II. Time series extraction - Seed </font>

# In[ ]:


seed_timeseries={};seed_timeseries_mean={};seed_timeseries_pc1={}
for seed_name in seed_names:
    seed_timeseries[seed_name],seed_timeseries_mean[seed_name],seed_timeseries_pc1[seed_name]=seed2voxels.extract_data(img=data_seed, mask= mask_seed[seed_name], timeseries_txt=ts_seed_txt[seed_name],
                                                                run="extract",n_jobs=8) # run the analyse for target voxels


# ## <font color=#B2D732> <span style="background-color: #4424D6"> Correlation analysis

# In[ ]:


output_img={};

for seed_name in seed_names:
    output_img[seed_name]=[]
    output_img[seed_name]=config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/' + seed_name + '/' + target_name[0].split('_')[0]+ '_fc_maps/' + str(len(config['list_subjects'])) + 'subjects_seed_' + seed_name.split('_')[-1] + '_s_BP_newdenoising'

    seed2voxels.correlation_maps(seed_timeseries_mean[seed_name],target_timeseries,mask=mask_target,output_img=output_img[seed_name],Fisher=True,partial=False,n_jobs=8)