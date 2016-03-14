#import use_ngmca as use
import pyfits as pf
import numpy as np
import matplotlib.pyplot as plt
from MuSCADeT import MCA as MC
import scipy.stats as sc
from MuSCADeT  import colour_subtraction as cs
import scipy.signal as scp


cube = pf.open('Simu_2744/Cube.fits')[0].data
#cube = np.multiply(cube.T, 1./np.sum(np.sum(cube,1),1)).T

num,n,n = np.shape(cube)


hdus = pf.PrimaryHDU(cube)
lists = pf.HDUList([hdus])
lists.writeto('Simu_2744/All_real.fits', clobber=True)

Aprior = pf.open('./Simu_2744/Estimated_A.fits')[0].data


##param
mom = 'mom'
positivity =False
reweighting ='none'
pca = 'PCA'
wmode = 'add'
soft = False
npca = 32
n = 800
#[145,255]

S,A = MC.mMCA(cube, Aprior, 5,n,threshmode = mom,  PCA=[2,11], harder =0, alpha = [142,256], npca = npca, pos = positivity,mode=pca)


pen = reweighting
hdus = pf.PrimaryHDU(S)
lists = pf.HDUList([hdus])
lists.writeto('Simu_2744/Sources_'+str(n)+'_'+pen+'.fits', clobber=True)

hdus = pf.PrimaryHDU(A)
lists = pf.HDUList([hdus])
lists.writeto('Simu_2744/Estimated_A.fits', clobber=True)

cs.make_colour_sub('Simu_2744/Sources_'+str(n)+'_'+pen+'.fits',
                   'Simu_2744/Estimated_A.fits',
                   './Simu_2744/All_real.fits','big_'+str(n)+'_'+pen, cuts = ['0','0.25','0','0.15','0','0.02','0','0.1','0','0.1'], prefix = './Simu_2744/')

plt.show()
