
% Second level one sample t test for SpPark dataset
% CL Septembre 2020, landelle.caroline@gmail.com // caroline.landelle@mcgill.ca

% Toolbox required: Matlab, SPM12

%% 
function Stat=Stat_SPPark(inputdir, subfolder, subjects_label, mask, seed_name,outputDir)

%______________________________________________________________________
%% Initialization 
%______________________________________________________________________
SPM_Dir='/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spm12/';
addpath(SPM_Dir) % Add SPM12 to the path
mask
for sbj=1:length(subjects_label) ;
subject_name=char(subjects_label(sbj));
fullfile(inputdir, ['sub-',subject_name]);
f{sbj,:} = spm_select('FPList',fullfile(inputdir, ['sub-',subject_name],subfolder), ['^wsub.*', 'seed_',seed_name,'.*_correlation_z_BP_ts_spm.nii$'])
end

matlabbatch{1}.spm.stats.factorial_design.dir = {outputDir};
matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = f;
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});

matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;

matlabbatch{1}.spm.stats.factorial_design.masking.em = {mask};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
spm_jobman('initcfg');

matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('Factorial design specification: SPM.mat File' , substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 1;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'Positive';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1];
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{3}.spm.stats.con.spmmat(2) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'Negative';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1];
matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

matlabbatch{3}.spm.stats.con.delete = 0;

spm_jobman('run',matlabbatch);
Stat='Analyse done'
clear matlabbatch
end   