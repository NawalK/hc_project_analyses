
class FC_Parcellation:
    '''
    The FC_Parcellation class is used to perform the parcellation of a specific roi
    based on the FC profiles of each of its voxel
    
    Attributes
    ----------
    config : dict
    signal: str
        type of signal ('raw' for bold or 'ai' for deconvoluted)
    seed_names: 2D array
        name of the seeds (ex: ['spinalcord_seed1','spinalcord_seed2'])
    target_name: 2D array
        name of the target structure (ex: ['brain_GM'])
    '''
    
    def __init__(self, config):
        self.config = config # Load config info
       
    