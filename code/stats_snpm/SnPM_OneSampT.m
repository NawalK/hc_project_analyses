% CL June 2023, landelle.caroline@gmail.com // caroline.landelle@mcgill.ca

% Toolbox required: Matlab, SPM12

%% 
function SnPM_OneSampT=SnPM_OneSampT(permutation,t_thr,input_dir,tag_file,mask_filename,output_dir,output_filename)

%______________________________________________________________________
%% Initialization 
%______________________________________________________________________
SPM_Dir='/cerebro/cerebro1/dataset/bmpd/derivatives/thibault_test/code/toolbox/spm12/'
addpath(SPM_Dir); % Add SPM12 to the path

permutation
files=spm_select('ExtFPList',fullfile(input_dir),[tag_file,'.*s.nii$'])
for sbj=1:size(files,1) ;
f{sbj,:} = files(sbj,:);
end



%______________________________________________________________________
%% Stats
%______________________________________________________________________

matlabbatch{1}.spm.tools.snpm.des.OneSampT.DesignName = 'MultiSub: One Sample T test on diffs/contrasts';
matlabbatch{1}.spm.tools.snpm.des.OneSampT.DesignFile = 'snpm_bch_ui_OneSampT';
matlabbatch{1}.spm.tools.snpm.des.OneSampT.dir = {output_dir};

matlabbatch{1}.spm.tools.snpm.des.OneSampT.P = f

matlabbatch{1}.spm.tools.snpm.des.OneSampT.cov = struct('c', {}, 'cname', {});
matlabbatch{1}.spm.tools.snpm.des.OneSampT.nPerm = permutation;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.vFWHM = [6 6 6];
matlabbatch{1}.spm.tools.snpm.des.OneSampT.bVolm = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.ST.ST_U = t_thr;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.masking.tm.tm_none = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.masking.im = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.masking.em = {mask_filename};
matlabbatch{1}.spm.tools.snpm.des.OneSampT.globalc.g_omit = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.tools.snpm.des.OneSampT.globalm.glonorm = 1;
matlabbatch{2}.spm.tools.snpm.cp.snpmcfg(1) = cfg_dep('MultiSub: One Sample T test on diffs/contrasts: SnPMcfg.mat configuration file', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','SnPMcfg'));
matlabbatch{3}.spm.tools.snpm.inference.SnPMmat(1) = cfg_dep('Compute: SnPM.mat results file', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','SnPM'));
matlabbatch{3}.spm.tools.snpm.inference.Thr.Clus.ClusSize.CFth = NaN;
matlabbatch{3}.spm.tools.snpm.inference.Thr.Clus.ClusSize.ClusSig.FWEthC = 0.05;
matlabbatch{3}.spm.tools.snpm.inference.Tsign = 1;
matlabbatch{3}.spm.tools.snpm.inference.WriteFiltImg.name = output_filename;
matlabbatch{3}.spm.tools.snpm.inference.Report = 'MIPtable';

spm_jobman('initcfg');
spm_jobman('run',matlabbatch);

SnPM_OneSampT='One Sample T test done'
