from cmath import nan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import center_of_mass,label,find_objects
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter 



def compute_similarity(config, data1, data2, thresh1=2, thresh2=2, mask1=None, mask2=None, method='Dice', match_compo=True, plot_results=False,save_results=False,verbose=True):
    ''' Compute the spatial similarity between two sets of 3D components
        
        Inputs
        ----------
        config : dict
            Contains information regarding data savings, etc.
        data1, data2 : 4D arrays 
            Contain the maps of the components to analyze (3D x nb of components) 
        thresh1, thresh2 : float
            Lower threshold value to binarize components (default = 2)
        mask1, mask2: 3D arrays
            Only needed for method='Cosine' (default = None)
            Will be used to mask the maps before computing the cosine similarity
        method : str
            Method to compute similarity (default = 'Dice')
                'Dice' to compute Dice coefficients (2*|intersection| / nb_el1 + nb_el2)
                'Euclidean distance abs' to compute the inverse absolute distance between the centers of mass (unit in voxel)
                'Euclidean distance' to compute the distance between the centers of mass (unit in voxel)
                'Cosine' to compute cosine similarity 
        match_compo : boolean
            Match & order components based on max similarity (default = True)
        save_results : str
            Defines whether results are saved or not (default = False)
        verbose : bool
            Indicates wheter information on what is being done are printed (default = True)
        
        Outputs
        ------------
        similarity_matrix : 3D array
            Matrix containing the similarity between pairs of components from the two datasets 
        orderX: 2D array
            Array containing the order of the first dataset after matching (x in the plotting)
        orderY: 2D array
            Array containing the order of the second dataset after matching (y in the plotting)
        '''

    if verbose == True:
        print(f"COMPUTING SIMILARITY WITH METHOD: {method}")

    # Number of components is equal to the max between the two sets
    k = np.max([data1.shape[3],data2.shape[3]]) # Save number of components for later use, shape 3 = number of components
    
    if method == 'Dice' or method == 'Overlap' or method == 'Euclidean distance': # Binarize data if needed
        data1_bin = np.where(data1 >= thresh1, 1, 0)
        data2_bin = np.where(data2 >= thresh2, 1, 0)
    elif method == 'Cosine': # Prepare structures to save vectorized maps if needed
        if mask1 is None or mask2 is None: # Check if masks have been given
            raise(Exception(f'The "Cosine" method requires masks as inputs!'))
        mask1_vec = np.reshape(mask1,(mask1.shape[0]*mask1.shape[1]*mask1.shape[2],1)) # Reshape masks
        mask2_vec = np.reshape(mask2,(mask2.shape[0]*mask2.shape[1]*mask2.shape[2],1)) 
        if np.count_nonzero(mask1_vec) != np.count_nonzero(mask2_vec):
            raise(Exception(f'The "Cosine" method can only be used for data using the same masks!'))
        data1_vec = np.zeros((data1.shape[0]*data1.shape[1]*data1.shape[2],k))
        data2_vec = np.zeros((data2.shape[0]*data2.shape[1]*data2.shape[2],k))
        data1_masked = np.zeros((np.count_nonzero(mask1_vec),k))
        data2_masked = np.zeros((np.count_nonzero(mask2_vec),k))    
        

    if verbose == True:
        print(f"...Compute similarity between pairs of components")
    similarity_matrix = np.zeros((k,k))
        
    for k1 in range(0,k):
        if method == 'Cosine': # Reshape as vector & mask if needed 
            data1_vec[:,k1] = np.reshape(data1[:,:,:,k1],(data1.shape[0]*data1.shape[1]*data1.shape[2],))
            data1_masked[:,k1] = data1_vec[np.flatnonzero(mask1_vec),k1]
        for k2 in range(0,k):
            if method == 'Dice' or method == 'Overlap':
                # For the intersection, we multiply the two binary maps and count the number of elements
                if k1 < data1.shape[3] and k2 < data2.shape[3]: # If the element exist in both datasets, we compute the similarity
                    nb_el_inters = np.sum(np.multiply(data1_bin[:,:,:,k1], data2_bin[:,:,:,k2])) 
                    nb_el_1 = np.sum(data1_bin[:,:,:,k1])
                    nb_el_2 = np.sum(data2_bin[:,:,:,k2])
                    if method == 'Dice':
                        similarity_matrix[k1,k2] = 2*nb_el_inters / (nb_el_1+nb_el_2)
                    elif method == 'Overlap':
                        if nb_el_1 > nb_el_2:
                            similarity_matrix[k1,k2] = nb_el_inters / (nb_el_2)
                        elif nb_el_2 > nb_el_1:
                            similarity_matrix[k1,k2] = nb_el_inters / (nb_el_1)
                        elif nb_el_2 == nb_el_1:
                            similarity_matrix[k1,k2] = 2*nb_el_inters / (nb_el_1+nb_el_2)
                            print("the two cluster are equal, 'Dice' methods was applied instead of 'Dice_smaller'")
                            
                        
                else: # Else, we just set it to -1
                    similarity_matrix[k1,k2] = -1
            elif method == 'Euclidean distance' or method == 'Euclidean distance abs' :
                if k1 < data1.shape[3] and k2 < data2.shape[3]: # If the element exist in both datasets, we compute the similarity
                    # Label data to find the different clusters
                    lbl1 = label(data1_bin[:,:,:,k1])[0]
                    lbl2 = label(data2_bin[:,:,:,k2])[0]
    
                    # We calculate the center of mass of the largest clusters
                    cm1 = center_of_mass(data1_bin[:,:,:,k1],lbl1,Counter(lbl1.ravel()).most_common()[1][0])
                    cm2 = center_of_mass(data2_bin[:,:,:,k2],lbl2,Counter(lbl2.ravel()).most_common()[1][0])

                    if method == 'Euclidean distance abs':
                        # inverse of the euclidean distance between CoG
                        #similarity_matrix[k1,k2]=1/(np.mean(np.abs(np.array(cm1)-np.array(cm2)))) 
                        similarity_matrix[k1,k2] = 1/(np.mean(np.abs([float(cm1[1])-float(cm2[1]),float(cm1[2])-float(cm2[2])])))
                    
                    elif method == 'Euclidean distance':
                        similarity_matrix[k1,k2] = np.mean([float(cm1[1])-float(cm2[1]),float(cm1[2])-float(cm2[2])])
                else:
                    similarity_matrix[k1,k2] = -1
            elif method == 'Cosine':
                data2_vec[:,k2] = np.reshape(data2[:,:,:,k2],(data2.shape[0]*data2.shape[1]*data2.shape[2],)) # Vectorize
                data2_masked[:,k2] = data2_vec[np.flatnonzero(mask2_vec),k2]
                if k1 < data1.shape[3] and k2 < data2.shape[3]: # If the element exist in both datasets, we compute the similarity
                    similarity_matrix[k1,k2] = cosine_similarity(data1_masked[:,k1].reshape(1, -1), data2_masked[:,k2].reshape(1, -1))
                else:
                    similarity_matrix[k1,k2] = -1
            else:
                raise(Exception(f'The method {method} has not been implemented'))
        
    if match_compo == True:
        if verbose == True:
            print(f"...Ordering components based on maximum weight matching")
        orderX,orderY=scipy.optimize.linear_sum_assignment(similarity_matrix,maximize=True)
        # if the same composantes match
        similarity_matrix = similarity_matrix[:,orderY]
    else:
        orderY = np.array(range(0,k))

    # Plot similarity matrix
    if plot_results == True:
        sns.heatmap(similarity_matrix,linewidths=.5,square=True,cmap='YlOrBr',vmax=1,xticklabels=orderY+1,yticklabels=np.array(range(1,k+1)));

    # Saving result and figure if applicable
    if save_results == True:
        plt.savefig(config['output_dir'] + config['output_tag'] + '_k' + str(k) +'.png')
    
    if verbose == True:
        print(f"DONE!")

    return similarity_matrix, orderX, orderY


