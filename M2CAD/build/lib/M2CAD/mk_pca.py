import numpy as np


def mk_pca(curves, dec = 0):
    #Elements to make pca on are the columns of the input curves

    if np.sum(np.sum(dec)) != 0:
        E_N2 = dec
    else:
    
    #Computing covariance matrix
        cov = np.dot(np.transpose(curves), curves)
        print('svd going on')
    #Singular value decomposition
        [U,W,V] = np.linalg.svd(cov)
    #Computing Eigenvectors
    #a=np.dot(np.transpose(U),curves)
        E_N2 = np.dot(np.dot(curves,np.transpose(V)),np.diag(1/np.sqrt(W)))
    print('svd done')
    #Computing eigenvalues
    alpha = np.dot(np.transpose(E_N2),curves)

    return alpha, E_N2
    


def rec_pca(alpha, base, lim = 0):

    if lim != 0:
        alpha = alpha[:lim-1,:]
        base = base [:,:lim-1]
        
    rec = np.dot(base,alpha)

    return rec
