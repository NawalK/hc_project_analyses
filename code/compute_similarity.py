import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

def compute_similarity(config, data1, data2, thresh1=2, thresh2=2, method='Dice', match_compo=True, plot_results=False,save_results=False):
    ''' Compute the spatial similarity between two sets of 3D components
        
        Inputs
        ----------
        config : dict
            Contains information regarding data savings, etc.
        data1, data2 : 4D arrays 
            Contain the maps of the components to analyze (3D x nb of components) 
        thresh1, thresh2 : float
            Lower threshold value to binarize components (default = 2)
        method : str
            Method to compute similarity (default = 'Dice')
                'Dice' to compute Dice coefficients (2*|intersection| / nb_el1 + nb_el2) 
        match_compo : boolean
            Match & order components based on max similarity (default = True)
        save_results : str
            Defines whether results are saved or not (default = False)
        
        Outputs
        ------------
        similarity_matrix : 3D array
            Matrix containing the similarity between pairs of components from the two datasets 
        orderX: 2D array
            Array containing the order of the first dataset after matching (x in the plotting)
        orderY: 2D array
            Array containing the order of the second dataset after matching (y in the plotting)
        '''

    print(f"COMPUTING SIMILARITY WITH METHOD: {method}")

    # Check that datasets have equal number of components
    if data1.shape[3] == data2.shape[3]:
        k = data1.shape[3] # Save number of components for later use
        print(f"...Binarize data \n Threshold 1 = {thresh1} \n Threshold 2 = {thresh2}")
        data1_bin = np.where(data1 >= thresh1, 1, 0)
        data2_bin = np.where(data2 >= thresh2, 1, 0)

        print(f"...Compute similarity between pairs of components")
        similarity_matrix = np.zeros((k,k))
        
        for k1 in range(0,k):
            for k2 in range(0,k):
                if method == 'Dice':
                    # For the intersection, we multiply the two binary maps and count the number of elements
                    nb_el_inters = np.sum(np.multiply(data1_bin[:,:,:,k1], data2_bin[:,:,:,k2])) 
                    nb_el_1 = np.sum(data1_bin[:,:,:,k1])
                    nb_el_2 = np.sum(data2_bin[:,:,:,k2])
                    similarity_matrix[k1,k2] = 2*nb_el_inters / (nb_el_1+nb_el_2)
                else:
                    raise(Exception(f'The method {method} has not been implemented'))
        
        if match_compo == True:
            print(f"...Ordering components based on maximum weight matching")
            orderX,orderY=scipy.optimize.linear_sum_assignment(similarity_matrix,maximize=True)
            # if the same composantes match
            similarity_matrix = similarity_matrix[:,orderY]
        else:
            orderY = np.array(range(0,k))

        # Plot similarity matrix
        if plot_results == True:
            sns.heatmap(similarity_matrix,linewidths=.5,square=True,cmap='YlOrBr',xticklabels=orderY+1,yticklabels=np.array(range(1,k+1)));

        # Saving result and figure if applicable
        if save_results == True:
            plt.savefig(config['output_dir'] + config['output_tag'] + '_k' + str(k) +'.png')
    

    else:
        raise(Exception(f'The two datasets have different number of components ({data1.shape[3]} vs {data2.shape[3]}).'))
    
    print(f"DONE!")

    return similarity_matrix, orderX, orderY

