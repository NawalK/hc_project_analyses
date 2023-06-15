% List of open inputs
nrun = X; % enter the number of runs here
jobfile = {'/Users/carolinelandelle/mnt/bagot/bmpd/derivatives/HealthyControls_project/seed_to_voxels/2_second_level/SnPM_script_job.m'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(0, nrun);
for crun = 1:nrun
end
spm('defaults', 'PET');
spm_jobman('run', jobs, inputs{:});
