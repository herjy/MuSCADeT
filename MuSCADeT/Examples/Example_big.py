import pyfits as pf
import numpy as np
import matplotlib.pyplot as plt
from MuSCADeT import MCA
from MuSCADeT import pca_ring_spectrum as pcas
import scipy.stats as sc
from MuSCADeT import colour_subtraction as cs

## Openning data cube
cube = pf.open('Simu_big/Cube.fits')[0].data
num,n1,n2 = np.shape(cube)

## A for toy model
Aprior = pf.open('Simu_simple/Simu_A.fits')[0].data

## Input parameters
pca = 'PCA'     #Estimation of the mixing coefficients from PCA. If different from PCA it will use the array provided in Aprior
n = 100         #Number of iterations
nsig = 5        #Threshold in units of noise standard deviation
ns = 2          #Number of sources
angle = 5      #Resolution angle for the PCA colour estimation (start with 15 then adjust empirically)

## Running MuSCADeT
S,A,Chi = MCA.mMCA(cube, Aprior.T, nsig,n, PCA=[ns,angle], mode=pca)

hdus = pf.PrimaryHDU(S)
lists = pf.HDUList([hdus])
lists.writeto('Simu_big/Sources_'+str(n)+'.fits', clobber=True)

hdus = pf.PrimaryHDU(A)
lists = pf.HDUList([hdus])
lists.writeto('Simu_big/Estimated_A.fits', clobber=True)

cs.make_colour_sub('Simu_big/Sources_'+str(n)+'.fits',
                   'Simu_big/Estimated_A.fits',
                   './Simu_big/Cube.fits',
                   'big_'+str(n),
                   prefix = './Simu_big/',
                   cuts = ['-0.1','0.6','-0.05','0.3','-0.02','0.1'])

