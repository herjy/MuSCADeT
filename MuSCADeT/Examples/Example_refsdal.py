import pyfits as pf
import numpy as np
import matplotlib.pyplot as plt
from MuSCADeT import MCA
from MuSCADeT import pca_ring_spectrum as pcas
import scipy.stats as sc
from MuSCADeT import colour_subtraction as cs

## Openning data cube
cube = pf.open('./Simu_Refsdal_big/Cube.fits')[0].data
num,n,n = np.shape(cube)

## A for toy model
Aprior =pf.open('Simu_Refsdal_big/Estimated_A_PCA.fits')[0].data

## Input parameters
pca = 'noPCA'   #Estimation of the mixing coefficients from PCA. If different from PCA it will use the array provided in Aprior
n = 2000        #Number of iterations
nsig = 5        #Threshold in units of noise standard deviation
ns = 2          #Number of sources
angle = 50      #Resolution angle for the PCA colour estimation (start with 15 then adjust empirically)

## Running MuSCADeT
S,A = MCA.mMCA(cube, Aprior, nsig,n, PCA=[ns,angle], mode=pca, harder = 1)


for i in [1]:
    hdus = pf.PrimaryHDU(S)
    lists = pf.HDUList([hdus])
    lists.writeto('Simu_Refsdal_big/Sources_'+str(n)+'.fits', clobber=True)

    hdus = pf.PrimaryHDU(A)
    lists = pf.HDUList([hdus])
    lists.writeto('Simu_Refsdal_big/Estimated_A.fits', clobber=True)

    cs.make_colour_sub('Simu_Refsdal_big/Sources_'+str(n)+'.fits',
                       'Simu_Refsdal_big/Estimated_A.fits',
                       './Simu_Refsdal_big/Cube.fits',
                       'Refsdal_big_'+str(n),
                       prefix = './Simu_Refsdal_big/',
                       cuts = ['0','0.1','-0.002','0.06','-0.002','0.03'])

