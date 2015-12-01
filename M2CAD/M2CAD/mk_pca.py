"""@package M2CAD

"""

import numpy as np


def mk_pca(vectors, dec = 0):
    """
    Perfoms Principal Component Analysis of a set of vectors
    INPUTS:
        vectors: Set of vectors to be decomposed through PCA.
    OUTPUTS:
        alpha: PCA coefficients resulting of the decomposition of the vectors.
        EN_2: PCA basis set.
    OPTIONS:
        dec: if non zero, dec is used as a PCA basis to decompose the vectors. In this case,
        a simple projection is thus conducted instead of the PCA.
    """

    if np.sum(np.sum(dec)) != 0:
        E_N2 = dec
    else:
    
    #Computing covariance matrix
        cov = np.dot(np.transpose(vectors), vectors)
        print('svd going on')
    #Singular value decomposition
        [U,W,V] = np.linalg.svd(cov)
    #Computing Eigenvectors
    #a=np.dot(np.transpose(U),vectors)
        E_N2 = np.dot(np.dot(vectors,np.transpose(V)),np.diag(1/np.sqrt(W)))
    print('svd done')
    #Computing eigenvalues
    alpha = np.dot(np.transpose(E_N2),vectors)

    return alpha, E_N2
    


def rec_pca(alpha, base, lim = 0):
    """
    Reconstructs a signal in direct space from its PCA coefficients and the basis over which it has been decomposed.
    INPUTS:
        alpha: sets of PCA coefficients.
        basis: the basis over which the signal has been decomposed.
    OUTPUTS:
        rec: reconstructed signal
    OPTIONS:
        lim: if lim is set to non-zero value, lim is maximal number of coefficients used in the reconstruction.
    """

    if lim != 0:
        alpha = alpha[:lim-1,:]
        base = base [:,:lim-1]
        
    rec = np.dot(base,alpha)

    return rec
