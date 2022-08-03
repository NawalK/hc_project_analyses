Participants included in the project:
__________________________________________________
From STRATALS and BMPD dataset
*  BMPD particicpants infos: https://docs.google.com/spreadsheets/d/1rhOjRb7uVJnytoppKZosQ8Crcp4WHUK1/edit#gid=2128824184
* STRATALS pariticpants infos: https://mcgill-my.sharepoint.com/:x:/r/personal/julien_doyon_mcgill_ca/_layouts/15/Doc.aspx?sourcedoc=%7BB3D281A0-F35E-41A2-8A0A-E08257A0C7A6%7D&file=Participants_update.xlsx&action=default&mobileredirect=true&cid=f98110fb-c344-4e7b-a028-7e8541527245

BMPD_P028
BMPD_P030
BMPD_P033
BMPD_P043
BMPD_P047
BMPD_P050
BMPD_P057
BMPD_P097
BMPD_P099
BMPD_P100
BMPD_P105
BMPD_P108
BMPD_P109
STRATALS_P001
STRATALS_P003
STRATALS_P004
STRATALS_P005
STRATALS_P006
STRATALS_P007
STRATALS_P008
STRATALS_P009
STRATALS_P011
__________________________________________________
Preprocessed data:
__________________________________________________
BMPD: 'cerebro/cerebro1/dataset/bmpd/derivatives/spinal_processing/'
STRATALS: 'cerebro/cerebro1/dataset/stratals/derivatives/preprocessing/'

* BMPD infos: https://docs.google.com/spreadsheets/d/12EKgE6-lZ5C0OC_ec8EEtTKHIeUBmjOI8rhFVgoohhE/edit?skip_itp2_check=true&pli=1#gid=588518088
* STRATALS infos: https://mcgill-my.sharepoint.com/:x:/r/personal/julien_doyon_mcgill_ca/_layouts/15/Doc.aspx?sourcedoc=%7BE0DE1087-D6CA-44BE-AEFC-7FCDD8BEC969%7D&file=progress.xlsx&action=default&mobileredirect=true&cid=2e7c5822-f689-48a5-8d9f-c113f22440ca

__________________________________________________
Folder for the analyses:
__________________________________________________
'cerebro/cerebro1/dataset/bmpd/derivatives/Func_analyses/HealthyControls'/
* Codes: contain codes
* func_preparation:
    - codes: read the notebooks for more details about the outputs
    - 1_func_denoised : 4D denoised images
    - 2_func_in_template : 4D denoised images wrapped in template (EPI resolution) 
    - 3_func smoothed :4D denosoised in template images smoothed
    - 4_func concatenated: 4D denoised in template and smoothed images with brain and sc part concatenated
    
* ICA: Analyse ICA
* seed_to_voxels: Analyse seed to voxels
* templates:
    - MNI: brain templates
    - PAM50: sc templates