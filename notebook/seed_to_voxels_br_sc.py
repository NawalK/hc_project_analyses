#!/usr/bin/env python
# coding: utf-8

# ## <font color=#B2D732> <span style="background-color: #4424D6"> Imports

# In[1]:


import sys,json
import glob, os
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/code/")

sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.0.0")
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.0.0/scripts") 
from spinalcordtoolbox.utils.sys import run_proc

from seed_to_voxels import Seed2voxels




# Load config file ------------------------------------------------------------
with open('/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/hc_project_analyses/config/config_CL.json') as config_file: # the notebook should be in 'xx/notebook/' folder #config_proprio
    config = json.load(config_file) # load config file should be open first and the path inside modified
    config['list_subjects']=config["list_subjects_37"]
signal='raw'
    
seed_names=['spinalcord_C1','spinalcord_C2','spinalcord_C3','spinalcord_C4','spinalcord_C5','spinalcord_C6','spinalcord_C7','spinalcord_C8','spinalcord_C9'] # define seed names ex: 'spinalcord_ica-C4'
seed_names=['spinalcord_GM-C3toC4-D','spinalcord_GM-C3toC4-VI','spinalcord_GM-C3toC4-R','spinalcord_GM-C3toC4-L',]#,'spinalcord_GM-VI','spinalcord_GM-R','spinalcord_GM-L']#,'spinalcord_C1C8-LD','spinalcord_C1C8-LV','spinalcord_C1C8-RD','spinalcord_C1C8-RV',]
seed_names=['spinalcord_ica-C1','spinalcord_ica-C2-D','spinalcord_ica-C2-V','spinalcord_ica-C2C3-D','spinalcord_ica-C3-V','spinalcord_ica-C3C4',
           'spinalcord_ica-C4','spinalcord_ica-C4C5-L','spinalcord_ica-C5-L','spinalcord_ica-C5-R','spinalcord_ica-C6-R','spinalcord_ica-C6C7-V',
           'spinalcord_ica-C7-V','spinalcord_ica-C7C8','spinalcord_ica-C8']
seed_names=['spinalcord_GM-VL','spinalcord_GM-C7toC8','spinalcord_GM-C5toC6','spinalcord_GM-C3toC4','spinalcord_GM-C1toC2']#,'spinalcord_ica-C2-V','spinalcord_ica-C5-V']
target_name=['brain_mask'] # define structure target


# ### <font color=#4424D6> II. Select files </font>

# In[4]:


data_target=[];ts_target_txt=[];ts_seed_txt ={};data_seed=[];mask_seed={};mask_target=[]

#>>> Select data for extraction and target files -------------------------------------
for subject_name in config['list_subjects']:
    # select target files:
    mask_target.append(glob.glob(config["main_dir"] + config["masks"][target_name[0].split('_')[0]])[0])
    
    # select data for extraction:
    data_seed.append(glob.glob(config["coreg_dir"] + 'sub-'+ subject_name +'/'+seed_names[0].split('_')[0]+'/*' + config["coreg_tag"][seed_names[0].split('_')[0]] +'*')[0])
    data_target.append(glob.glob(config["coreg_dir"] + 'sub-'+ subject_name +'/'+target_name[0].split('_')[0]+'/*' + config["coreg_tag"][target_name[0].split('_')[0]] +'*.gz')[0])
    
    # Output filename (timeseries)
    ts_target_txt.append(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+target_name[0]+'/timeseries/sub_' + subject_name + '_mask_' + target_name[0] + '_timeseries')
    
#>>> Select seed files and ts -------------------------------------
for seed_name in seed_names:
    mask_seed[seed_name]=[]
    ts_seed_txt[seed_name]=[]
    print(seed_name)
    for subject_name in config['list_subjects']:
        mask_seed[seed_name].append(glob.glob(config["main_dir"] + config["seed2vox_dir"] + '/masks/'+ seed_name + '.nii.gz')[0])
        ts_seed_txt[seed_name].append(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name+'/timeseries/sub_' + subject_name + '_mask_' + seed_name.split('_')[-1] + '_timeseries')

    

#>>> create output directory if needed -------------------------------------
for seed_name in seed_names:
    if not os.path.exists(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name):
        os.mkdir(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name)
        os.mkdir(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name+'/timeseries/') # folder to store timeseries extraction
        os.mkdir(config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/'+seed_name+'/brain_fc_maps/') # folder to store maps of FC

    


# ### <font color=#4424D6> III. Initialize class based on this </font>

# In[5]:


for seed_name in seed_names:
    seed2voxels=Seed2voxels(config,signal,seed_name,target_name) # initialize the function


# ## <font color=#B2D732> <span style="background-color: #4424D6"> B/ Data extraction </font></span>
# ### <font color=#4424D6> I. Time series extraction - Target </font>
# 

# In[6]:


target_timeseries,target_timeseries_mean,target_timeseries_pc1=seed2voxels.extract_data(img=data_target, mask=mask_target, timeseries_txt=ts_target_txt,
                                                                run="load",n_jobs=8,smoothing_fwhm=[6,6,6]) # run the analyse for target voxels


# ### <font color=#4424D6> II. Time series extraction - Seed </font>

# In[7]:


seed_timeseries={};seed_timeseries_mean={};seed_timeseries_pc1={}
for seed_name in seed_names:
    seed_timeseries[seed_name],seed_timeseries_mean[seed_name],seed_timeseries_pc1[seed_name]=seed2voxels.extract_data(img=data_seed, mask= mask_seed[seed_name], timeseries_txt=ts_seed_txt[seed_name],
                                                                run="load",n_jobs=8) # run the analyse for target voxels


# ## <font color=#B2D732> <span style="background-color: #4424D6"> C/ Correlation analysis

# In[8]:


output_img={};

for seed_name in seed_names:
    output_img[seed_name]=[]
    output_img[seed_name]=config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/' + seed_name + '/' + target_name[0].split('_')[0]+ '_fc_maps/' + str(len(config['list_subjects'])) + 'subjects_seed_' + seed_name.split('_')[-1] + '_s'

    #seed2voxels.correlation_maps(seed_timeseries_mean[seed_name],target_timeseries,mask_files=mask_target,output_img=output_img[seed_name],Fisher=True,partial=False,save_maps=True,redo=False,n_jobs=8)


# ## <font color=#B2D732> <span style="background-color: #4424D6"> D/ Mutual information

# In[9]:


output_img={};

for seed_name in seed_names:
    output_img[seed_name]=[]
    output_img[seed_name]=config['main_dir'] + config['seed2vox_dir'] + '/1_first_level/' + seed_name + '/' + target_name[0].split('_')[0]+ '_fc_maps/mMI_' + str(len(config['list_subjects'])) + 'subjects_seed_' + seed_name.split('_')[-1] + '_s'

    
    mi=seed2voxels.mutual_info_maps(seed_timeseries_mean[seed_name],
                                    target_timeseries,
                                    mask_files=mask_target,
                                    output_img=output_img[seed_name],
                                    save_maps=True,
                                    smoothing_output=[6,6,6],redo=False, n_jobs=8)


