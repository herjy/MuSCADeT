import pyfits as pf
import numpy as np
import matplotlib.pyplot as plt
from MuSCADeT import MCA
from MuSCADeT import pca_ring_spectrum as pcas
import scipy.stats as sc
from MuSCADeT import colour_subtraction as cs

## Openning data cube
cube = pf.open('./Simu_nottoosimple/Cube.fits')[0].data
num,n,n = np.shape(cube)

## A for toy model
Aprior = pf.open('Simu_nottoosimple/Simu_A.fits')[0].data


## Input parameters
pca = 'PCA'     #Estimation of the mixing coefficients from PCA. If different from PCA it will use the array provided in Aprior
n = 100         #Number of iterations
nsig = 5        #Threshold in units of noise standard deviation
ns = 2          #Number of sources
angle = 25      #Resolution angle for the PCA colour estimation (start with 15 then adjust empirically)

## Running MuSCADeT
S,A = MCA.mMCA(cube, Aprior.T, nsig,n, PCA=[ns,angle], mode=pca)

## Writting results
hdus = pf.PrimaryHDU(S)
lists = pf.HDUList([hdus])
lists.writeto('Simu_nottoosimple/Sources_'+str(n)+'.fits', clobber=True)

hdus = pf.PrimaryHDU(A)
lists = pf.HDUList([hdus])
lists.writeto('Simu_nottoosimple/Estimated_A.fits', clobber=True)

## Plot residuals
cs.make_colour_sub('Simu_nottoosimple/Sources_'+str(n)+'.fits',
                   'Simu_nottoosimple/Estimated_A.fits',
                   './Simu_nottoosimple/Cube.fits',
                   prefix = './Simu_nottoosimple/')


