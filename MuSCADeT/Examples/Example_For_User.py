#import use_ngmca as use
import pyfits as pf
import numpy as np
import matplotlib.pyplot as plt
from MuSCADeT import MCA
from MuSCADeT import wave_transform as mw
import scipy.stats as sc
from MuSCADeT import colour_subtraction as cs

## Initialise your data with a cube with size (nb,n1,n2) where nb is the number of bands in your observations
cube = ##To be filed

hdus = pf.PrimaryHDU(cube)
lists = pf.HDUList([hdus])
lists.writeto('Simu_For_User/Cube.fits', clobber=True)

num,n,n = np.shape(cube)

## If A is unknown, set it to zero.
Aprior = 0 #pf.open('Simu_simple/Simu_A.fits')[0].data

## Input parameters
pca = 'PCA'     #Estimation of the mixing coefficients from PCA. If different from PCA it will use the array provided in Aprior
n = 100        #Number of iterations: increase if the separation is not good enough!
nsig = 5        #Threshold in units of noise standard deviation: Can be lowered down to 3 but 5 should be fine.
ns = 2          #Number of sources:
angle = 10      #Resolution angle for the PCA colour estimation (start with 15 then adjust empirically)
alpha = [0,0]   #If automated estimation of PCA coefficients fails, chose adequate alphas. See readme for more details
plot = False     #option to plot the PCA coefficients of the SEDs in the image. This option is usefull if one wants to make sure that SEDs have been correctly estimated. In automated mode, keep this option at False. In case the SEDs have to be refined, set plot to True, identify the features (alignements) on the plot that stand for different SEDs and use this to give values for alpha. (see readme.)

## Running MuSCADeT
S,A = MCA.mMCA(cube, Aprior.T, nsig,n, PCA=[ns,angle], mode=pca, alpha = [0,0])

## MuSCADeT estimates ns source, which means, variable S contains ns images
## Saves the sources in a fits file
hdus = pf.PrimaryHDU(S)
lists = pf.HDUList([hdus])
lists.writeto('Simu_For_User/Sources_'+str(n)+'.fits', clobber=True)

## Saves the estimated mixing matrix, A, in a fits file
hdus = pf.PrimaryHDU(A)
lists = pf.HDUList([hdus])
lists.writeto('Simu_For_User/Estimated_A.fits', clobber=True)

## This command shows the result of the separation in various formats:
## The command needs : A fits file with the extracted source and mixing coefficients (MuSCADeT's outputs)
##                     A fits file with the original data
##                     An optional prefix for file names
##                     Cuts: optional, these are the lower and upper cuts for Red, Green, Blue bands and for the sources to be displayed
## This command will show in DS9 the sources as extrated by MuSCADeT, the original images in RGB, the subtraction of blue (or red) component from the original data, the subtraction of red (or bleu) component from the original data and finally the residuals. If the separation is successful, the residuals should present only nonise.
cs.make_colour_sub('Simu_For_User/Sources_'+str(n)+'.fits',
                   'Simu_For_User/Estimated_A.fits',
                   './Simu_For_User/Cube.fits',
                   prefix = './Simu_For_User/',
                   cuts = ['-0.1','0.6','-0.05','0.3','-0.02','0.1', '0','0.5'])
