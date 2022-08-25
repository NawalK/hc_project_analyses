
% Second level one sample t test for SpPark dataset
% CL Septembre 2020, landelle.caroline@gmail.com // caroline.landelle@mcgill.ca

% Toolbox required: Matlab, SPM12

%% 
function Stat=Paired_t_Test(inputdir1,inputdir2,seed_name1,seed_name2,outputDir,config)

%______________________________________________________________________
%% Initialization 
%______________________________________________________________________
config.SPM_Dir
addpath(config.SPM_Dir) % Add SPM12 to the path

matlabbatch{1}.spm.stats.factorial_design.dir = {outputDir};
fullfile(inputdir1)
f1=spm_select('ExtFPList',fullfile(inputdir1), ['.*.nii']);
f2=spm_select('ExtFPList',fullfile(inputdir2), ['.*.nii']);
char(char(f1(1,:)),char(f2(1,:)));
spm_select('FPList','/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project//seed_to_voxels/1_first_level/spinalcord_C5-RV/brain_fc_maps/22subjects_seed_C5-RV_s_BP_zcorr.nii',1)



{spm_select('ExtFPList',fullfile('/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/seed_to_voxels/1_first_level/spinalcord_C5-RV/brain_fc_maps/'), ['22subjects_seed_C5-RV_s_BP_zcorr.nii',1])
spm_select('ExtFPList',fullfile('/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/seed_to_voxels/1_first_level/spinalcord_C5-RD/brain_fc_maps/'), ['22subjects_seed_C5-RD_s_BP_zcorr.nii',1])}

for sbj=1:length(config.list_subjects) ;
subject_name=char(config.list_subjects(sbj));
%matlabbatch{1}.spm.stats.factorial_design.des.pt.pair(sbj).scans = {char(char(f1(1,:)),char(f2(1,:)))}
end

matlabbatch{1}.spm.stats.factorial_design.des.pt.pair(1).scans = {spm_select('ExtFPList',fullfile('/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/seed_to_voxels/1_first_level/spinalcord_C5-RV/brain_fc_maps/'), ['22subjects_seed_C5-RV_s_BP_zcorr.nii',1])
spm_select('ExtFPList',fullfile('/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/seed_to_voxels/1_first_level/spinalcord_C5-RD/brain_fc_maps/'), ['22subjects_seed_C5-RD_s_BP_zcorr.nii',1])}

matlabbatch{1}.spm.stats.factorial_design.des.pt.pair(2).scans = {spm_select('ExtFPList',fullfile('/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/seed_to_voxels/1_first_level/spinalcord_C5-RV/brain_fc_maps/'), ['22subjects_seed_C5-RV_s_BP_zcorr.nii',2])
spm_select('ExtFPList',fullfile('/cerebro/cerebro1/dataset/bmpd/derivatives/HealthyControls_project/seed_to_voxels/1_first_level/spinalcord_C5-RD/brain_fc_maps/'), ['22subjects_seed_C5-RD_s_BP_zcorr.nii',2])}


matlabbatch{1}.spm.stats.factorial_design.des.pt.gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.pt.ancova = 0;
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;

matlabbatch{1}.spm.stats.factorial_design.masking.em = {};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
spm_jobman('initcfg');

matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File' , substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 1;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Seed1_Seed2';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 -1];
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{3}.spm.stats.con.spmmat(2) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'Seed2-Seed1';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 1];
matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

matlabbatch{3}.spm.stats.con.spmmat(3) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{3}.tcon.name = 'Main effect';
matlabbatch{3}.spm.stats.con.consess{3}.tcon.weights = [1 1];
matlabbatch{3}.spm.stats.con.consess{3}.tcon.sessrep = 'none';

matlabbatch{3}.spm.stats.con.delete = 0;

spm_jobman('run',matlabbatch);
Stat='Analyse done'
clear matlabbatch
end   