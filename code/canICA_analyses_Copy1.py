import shutil, json, glob

from scipy import linalg
from sklearn.utils.extmath import svd_flip
from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_random_state
from sklearn.decomposition import fastica
from operator import itemgetter
from scipy.stats import scoreatpercentile
import numpy as np
import fnmatch, os,sys
from nilearn.maskers import NiftiMasker
from nilearn.image import iter_img
from nilearn import image
from joblib import Parallel, delayed
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.0.0")
sys.path.append("/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spinalcordtoolbox-5.0.0/scripts")
from spinalcordtoolbox.utils.sys import run_proc


class ICA:
    
    def __init__(self,inputs,structures_ana,dataset,config):
        '''
        inputs: list
            structure to analyse shoud be "brain" or "spinalcord" or "brain_spinalcord" 
  
        structures_ana: str
            structure to analyse shoud be "brain" or "spinalcord" or "brain_spinalcord" 
        config : dict
            Contains information regarding subjects, paths, etc.
        input_dir: str
            path of the input directory


        '''
    
        self.structures_ana=structures_ana # name of the structure.s to analyze
        self.config = config # load config inf
        self.dataset=dataset # load config inf
        if '_' in self.structures_ana: # if the string contains '_' then 2 structures will be processed
            self.structures=[self.structures_ana.split('_')[0],self.structures_ana.split('_')[1]] #
            print('Analyse will be run on ' + self.structures[0] + ' and ' + self.structures[1] + ' structures simultaneously')
        else: # else only one structure
            self.structures=self.structures_ana[0]
            print('Analyse will be run on ' + self.structures +' structure')
            print('Mask: ' + self.config["main_dir"] + self.config["masks"][dataset][self.structures])
       
        # Load inputs for each individual and each structure
        self.files_func={};self.func_allsbj={}
        if '_' in self.structures_ana:
            for structure in self.structures:
                print(structure)
                self.files_func[structure]={};self.func_allsbj[structure]=[]
                for sbj_nb in range(0,len(self.config["list_subjects"][dataset])):
                    subject_name=self.config["list_subjects"][dataset][sbj_nb]
                    self.files_func[structure][subject_name]=inputs[sbj_nb]
        else:
            print(self.structures)
            self.files_func[self.structures]={};self.func_allsbj[self.structures]=[]
            for sbj_nb in range(0,len(self.config["list_subjects"][dataset])):
                subject_name=self.config["list_subjects"][dataset][sbj_nb]
                self.files_func[self.structures][subject_name]=inputs[sbj_nb]
       
                
            
        #output dir--------------------------------------------
        self.analyse_dir=self.config["main_dir"] + self.config["data"][self.dataset]["ica"][self.structures]["dir"]+'/' +self.dataset+'/'+ self.structures + '/' + '/K_' + str(self.config["n_comp"])
        if '_' in self.structures_ana:
            if not os.path.exists(self.analyse_dir):
                os.mkdir(self.analyse_dir)
                for structure in self.structures:
                    os.mkdir(self.analyse_dir +'/'+structure)
                    os.mkdir(self.analyse_dir +'/'+structure +'/comp_raw/')
                    os.mkdir(self.analyse_dir +'/'+structure + '/comp_zscored/')
                    os.mkdir(self.analyse_dir +'/'+structure + '/comp_bin/')
                    os.mkdir(self.analyse_dir +'/'+structure  + '/comp_indiv/')  
                    if not os.path.exists(self.analyse_dir +'/'+structure  + '/subject_data/'):
                        os.mkdir(self.config["main_dir"] + self.config["data"][self.dataset]["ica"][self.structures]["dir"] +'/'+  '/subject_data/')
                
            
        else:
            if not os.path.exists(self.analyse_dir):
                os.mkdir(self.analyse_dir)
                os.mkdir(self.analyse_dir + '/comp_raw/')
                os.mkdir(self.analyse_dir + '/comp_zscored/')
                os.mkdir(self.analyse_dir + '/comp_bin/')
                os.mkdir(self.analyse_dir +'/comp_indiv/')
                if not os.path.exists(self.config["main_dir"] + self.config["data"][dataset]["ica"][self.structures_ana[0]]["dir"]+'/' +self.dataset+ '/'+ self.structures +  '/subject_data/'):
                        os.mkdir(self.config["main_dir"] + self.config["data"][self.dataset]["ica"][self.structures_ana[0]]["dir"]+'/' +self.dataset+ '/'+ self.structures +  '/subject_data/')          
        
        
        # copy the config file in the output dir
        with open(self.analyse_dir + '/ICA_anayses_config.json', 'w') as fp:
            json.dump(config, fp)


    def get_data(self,t_r=None,run=None,n_jobs=1):
        '''
        Extract voxel values inside a mask for each individual
        Attributes
        ----------
        mask: str
        image mask filename
          
        return
        ----------
        data_all_structure: list 
        list lenght:  nb of subject
        Array inside the list = values inside each voxel(volumes x total nb voxels)
    
        '''
        
        data_all={};data_all[self.structures]=[];
        data_sbj={};self.nifti_masker={};
        if run==None:
            print('Please choose the methods: run=load or run=extract')
            
        if t_r==None:
            print('Please enter the time repetition, e.g. t_r=1.5')
      
        if '_' in self.structures_ana:
            structures=self.structures
        else:
            structures=self.structures_ana
        for structure in structures:
            
        #Extract the data inside the mask and create a vector
        #---------------------------------------------------------
            self.nifti_masker[structure]= NiftiMasker(mask_img=self.config["main_dir"] + self.config["masks"][self.dataset][structure],
                                 t_r=t_r,low_pass=None,high_pass=None,standardize=False,smoothing_fwhm=None).fit() #Extract the data inside the mask and create a vector
             
            data_sbj[structure]={};self.data_txt={}
            for subject_name in self.config["list_subjects"][self.dataset]:
                self.data_txt[subject_name]=self.config["main_dir"] + self.config["data"][self.dataset]["ica"][structure]["dir"] + self.dataset + '/'+ structure +'/subject_data/' + '/sub-' + subject_name +'_'+structure+'_data.txt'
                
        
            if run=='load':
                data_sbj[structure]=Parallel(n_jobs=n_jobs)(delayed(np.loadtxt)(self.data_txt[subject_name]) for subject_name in self.config["list_subjects"][self.dataset])
    
            if run=='extract':
                data_sbj[structure]=Parallel(n_jobs=n_jobs)(delayed(self._extract_data)(subject_name,structure)
                                       for subject_name in self.config["list_subjects"][self.dataset])
        
        # Group individual data in a main dictionnary
        #---------------------------------------------------------                    
        # if more than one structure concatenate the arrays of the two structures for each individual :
        if '_' in self.structures_ana:
            data_sbj[self.structures_ana]={};data_all[self.structures_ana]=[]
            for subject_nb in range(0,len(self.config["list_subjects"][self.dataset])):
                data_sbj[self.structures_ana][subject_nb]=np.concatenate([data_sbj[self.structures[0]][subject_nb],data_sbj[self.structures[1]][subject_nb]],axis=1)# shape: n_volumes, n_voxels(brain + sc)
            data_all_structure=data_sbj[self.structures_ana]
        else:
            data_all_structure=data_sbj[self.structures]
        
        return data_all_structure
    
    def _extract_data(self,subject_name,structure):
        print(structure)
        data_indiv= self.nifti_masker[structure].transform(self.files_func[structure][subject_name]) # array for one subject, shape: n_volumes,n_voxels
        np.savetxt(self.data_txt[subject_name], data_indiv, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)  
        print(">> data extraction done <<")

        return data_indiv

        
    def indiv_PCA(self,data_all_structure,save_indiv_img=False):
        '''
        We separate observation noise from subject-specific patterns through principal component analysis (PCA).   
        The principal components explaining most of the variance for a given subject's data set form the patterns of interest, while the tail of the spectrum is considered as observation noise.  Specifically, for each subject, we
        use a singular value decomposition (SVD) of the individual data matrix
        Attributes
        ----------
        data_all_structure: list
            list(2D array):values inside each voxel(volumes x nb voxels)
        
        
        return
        ----------
        reducedata_all: array
        (n_comp_PCA_indiv * n_subjects,n_voxels)
         
        References:
        [1] Nilearn toolbox: https://github.com/nilearn/nilearn/blob/9ddfa7259de3053a5ed6655cd662e115926cf6a5/nilearn/decomposition/base.py#L85*

        '''
        n_comp_pca=2*self.config["ica_ana"]["n_comp"]#self.config["ICA_params"]["n_comp_PCA_indiv"]  #Number of composent for the PCAs at individual level
        reducedata_all=[]
        for sbj_nb in range(0,len(self.config["list_subjects"][self.dataset])):
            #Dimensionality reduction using truncated SVD 
            U, S, V = linalg.svd(data_all_structure[sbj_nb].T, full_matrices=False) # transpose the matrice of data in voxels x volumes
            U, V = svd_flip(U, V) # # flip eigenvectors' sign to enforce deterministic output
            # The "copy" are there to free the reference on the non reduceddata, and hence clear memory early
            U = U[:, :n_comp_pca].copy(); #The 1D projection axis with the maximum variance preserved
            S = S[:n_comp_pca]; #The maximum variance preserving axis perpendicular to previous. 
            V = V[:n_comp_pca].copy() #he third axis is automatically the one which is perpendicular to first two.
            U = U.T.copy()
            U = U * S[:, np.newaxis]
            if sbj_nb==0:
                reducedata_all = U
            else: 
                reducedata_all=np.concatenate([reducedata_all, U]) # concatenation of reduce data (n_comp*n_sbj,n_voxels)
            
        if save_indiv_img== True:
            if '_' in self.structures_ana:
                structures=self.structures
            else:
                structures=self.structures_ana
            for structure in structures:
                j=0
                for sbj_nb in range(0,len(self.config["list_subjects"][self.dataset])):
                    subject_name=self.config["list_subjects"][self.dataset][sbj_nb]
                    
                    for i in range(j,j+n_comp_pca):
                        j=i
                        
                    j=+i+1
                    components_img = self.nifti_masker[structure].inverse_transform(reducedata_all[j-n_comp_pca:j]) # transform the components in nifti
                    components_img.to_filename(self.analyse_dir + '/comp_indiv/sub-' + subject_name +'_comp_ICA.nii.gz') #save the n components for each subjects
                    print("- 4D image create for sbj  " +subject_name)
        print(">> Individual PCA done <<")
        return reducedata_all

    def get_CCA(self,reducedata_all):
        '''
        We use a generalization of canonical correlation analysis (CCA).
        Canonical Correlation is a procedure for assessing the relationship between variables. 
        Attributes
        ----------
        reducedata_all
        save_indiv_img: Boolean
            If the individual composants needs to be saved
        
        return
        ----------
        components_: array
        (n_voxels,n_comp)
        
        References:
        [1] Nilearn toolbox https://github.com/nilearn/nilearn/blob/9ddfa7259de3053a5ed6655cd662e115926cf6a5/nilearn/decomposition/multi_pca.py#L14*
        '''
        n_comp_cca=self.config["ica_ana"]["n_comp"]
        S = np.sqrt(np.sum(reducedata_all** 2, axis=1)) #Calculate the root sum square = canonical root or variate
        S[S == 0] = 1 # if one value is equale to 0 change it by 1
        reducedata_all /= S[:, np.newaxis] # divide data by canonical root (proportion of variance)
        components_, variance_, _=randomized_svd(reducedata_all.T, n_components=n_comp_cca,transpose=True,n_iter=3,random_state=None) #SVD is equivalent to standard CCA
        reducedata_all *= S[:, np.newaxis] # # Untransform the original reduced data

        

        return components_

    def get_ICA(self,components_):
    
        '''
        source separation using spatial ICA on subspace
        Attributes
        ----------
        components_: array
        (n_voxels,n_comp_PCA_indiv * n_subjects)
        
        iter: int
        The number of times the fastICA algorithm is restarted
    

        return
        ----------

        References:
        [1] Nilearn toolbox https://github.com/nilearn/nilearn/blob/9ddfa7259de3053a5ed6655cd662e115926cf6a5/nilearn/decomposition/multi_pca.py#L14*
        '''
        self.iter=self.config["ica_ana"]["iter"]
        components_final={};components_final_z={}
        random_state=None
        ratio=1.#float(15)

        random_state = check_random_state(random_state)
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.iter)
        components_.astype(np.float64) # to avoid NaN and inf values

        results = (fastica(components_, whiten=True, fun='cube',random_state=seed)
        for seed in seeds)

        ica_maps_gen_ = (result[2] for result in results)
        ica_maps_and_sparsities = ((ica_map,
                                    np.sum(np.abs(ica_map), axis=1).max())
                                   for ica_map in ica_maps_gen_)
        ica_maps, _ = min(ica_maps_and_sparsities, key=itemgetter(-1))
        
        #abs_ica_maps = np.abs(ica_maps)
        #percentile = 100. - (100. / len(ica_maps)) * ratio
        #threshold = scoreatpercentile(abs_ica_maps, percentile)
        #ica_maps[abs_ica_maps < threshold] = 0.
       
        
        components_= ica_maps.astype(components_.dtype)
        
        for component in components_:
            if component.max() < -component.min():
                component *= -1
        
        # Calculate the z-score for each composant -----------------------------------------
        components_z=np.zeros(shape=(components_.shape[0],components_.shape[1]),dtype='float')
        for i in range(0,len(components_.T)):
            med  = np.median(components_)
            components_z[:,i] = (components_[:,i]-med)/np.sqrt((np.sum((components_[:,i]-med)**2))/len(components_[:,i]))   # normalization 
                                                             
        if not '_' in self.structures_ana:
            components_final[self.structures]=components_
            components_final_z[self.structures]=components_z
        
        # For brain and spinal cord analyses split the component in two voxel matrices to be transform in two separates images in the next step
        if '_' in self.structures_ana:
            n_voxels={};nifti_masker={}
            for structure in self.structures:
                nifti_masker[structure]= NiftiMasker(mask_img=self.config["main_dir"] + self.config["masks"][structure], standardize=False,smoothing_fwhm=None).fit() #Extract the data inside the mask and create a vector
                n_voxels[structure]= nifti_masker[structure].fit_transform(self.config["main_dir"] + self.config[self.dataset]["masks"][structure]).shape[1] # number of voxels in the structure

                components_final[structure]=np.empty(shape=(n_voxels[structure],self.config["n_comp"]),dtype='float') # matrix of brain voxels 
                components_final_z[structure]=np.empty(shape=(n_voxels[structure],self.config["n_comp"]),dtype='float') # matrix of brain voxels 
            
            for voxel in range(0,n_voxels[self.structures[0]]):
                components_final[self.structures[0]][voxel,:]=components_[voxel,:]
                components_final_z[self.structures[0]][voxel,:]=components_z[voxel,:]
            for voxel in range(n_voxels[self.structures[0]],n_voxels[self.structures[0]]+n_voxels[self.structures[1]]):
                components_final[self.structures[1]][voxel-n_voxels[self.structures[0]],:]=components_[voxel,:]
                components_final_z[self.structures[1]][voxel-n_voxels[self.structures[0]],:]=components_z[voxel,:]
        
        print(">> Group ICA done <<")
        return  components_final, components_final_z
    
    def save_components(self,components_final,components_final_z):
        '''
        The iCA class is used to calculate CanICA in different structures (brain and/or spinalcord)
        Attributes
        ----------
        components_final : dict
            Contains information the ICAs
               
        outputs
        ----------
        components_filename: image
        '''
        nifti_masker={};
        
        if '_' in self.structures_ana:
            structures=self.structures
        else:
            structures=self.structures_ana
                
        for structure in structures:

            zimg=[]
        # Define output folders -----------------------------------------------------------------
            if not '_' in self.structures_ana:
                outputdir=self.analyse_dir  
            else:
                outputdir=self.analyse_dir +'/'+structure +'/'
                                                             
        # 1. Transform matrice of data into image nifti
        #-----------------------------------------------------------------
            nifti_masker[structure]= NiftiMasker(mask_img= self.config["main_dir"] + self.config["masks"][self.dataset][structure], standardize=False,smoothing_fwhm=None).fit() # mask
            
        #2. Save 4D image
        #----------------------------------------------------------------------
            components_img = nifti_masker[structure].inverse_transform(components_final[structure].T) #from matrice to nifti
            zcomponents_img = nifti_masker[structure].inverse_transform(components_final_z[structure].T) #check the component
        
            components4D_filename=outputdir +  '/comp_raw/CanICA_' + str(len(self.config["list_subjects"][self.dataset])) + 'sbj_'+ self.structures_ana[0] +'_'+structure +'_4D_K_'+ str(self.config["n_comp"]) + '.nii.gz' # filename of the 4D image
            zcomponents4D_filename=outputdir  + '/comp_zscored/zCanICA_' + str(len(self.config["list_subjects"][self.dataset])) + 'sbj_'+ self.structures_ana[0] +'_'+ structure + '_4D_K_'+ str(self.config["n_comp"]) + '.nii.gz'
            components_img.to_filename(components4D_filename)
            zcomponents_img.to_filename(zcomponents4D_filename)
        
        #3. Save 3D images
        #----------------------------------------------------------------------
            for i, cur_img in enumerate(iter_img(components_img)): #extract each composante of the image
                indiv_comp_img=outputdir + '/comp_raw/CanICA_' + str(len(self.config["list_subjects"][self.dataset])) + 'sbj_'+ self.structures_ana[0] +'_'+structure +'_k_' + str(i+1) + '.nii.gz' #filename
                cur_img.to_filename(indiv_comp_img) # save the image
                
            for i, zcur_img in enumerate(iter_img(zcomponents_img)):
                zindiv_comp_img=outputdir + '/comp_zscored/zCanICA_' + str(len(self.config["list_subjects"][self.dataset])) + 'sbj_'+ self.structures_ana[0] +'_'+structure +'_k_' + str(i+1) + '.nii.gz'
                zcur_img.to_filename(zindiv_comp_img)
        
        
        print(">> Components z-scored done <<")                                                                    
        return zcomponents4D_filename
    
    
    