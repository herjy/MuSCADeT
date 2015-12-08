"""@package MuSCADeT

"""

import numpy as np
import scipy.signal as cp
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sc

    
def wave_transform(img, lvl, Filter = 'Bspline', newwave = 1, convol2d = 0):
    """
    Performs starlet decomposition of an image
    INPUTS:
        img: image with size n1xn2 to be decomposed.
        lvl: number of wavelet levels used in the decomposition.
    OUTPUTS:
        wave: starlet decomposition returned as lvlxn1xn2 cube.
    OPTIONS:
        Filter: if set to 'Bspline', a bicubic spline filter is used (default is True).
        newave: if set to True, the new generation starlet decomposition is used (default is True).
        convol2d: if set, a 2D version of the filter is used (slower, default is 0).
        
    """
    mode = 'nearest'
    
    lvl = lvl-1
    sh = np.shape(img)
    if np.size(sh) ==3:
        mn = np.min(sh)
        wave = np.zeros([lvl+1,sh[1], sh[1],mn])
        for h in np.linspace(0,mn-1, mn):
            if mn == sh[0]:
                wave[:,:,:,h] = wave_transform(img[h,:,:],lvl+1, Filter = Filter)
            else:
                wave[:,:,:,h] = wave_transform(img[:,:,h],lvl+1, Filter = Filter)
        return wave
    n1 = sh[1]
    n2 = sh[1]
    
    if Filter == 'Bspline':
        h = [1./16, 1./4, 3./8, 1./4, 1./16]
    else:
        h = [1./4,1./2,1./4]
    n = np.size(h)
    h = np.array(h)
    
    if n+2**(lvl-1)*(n-1) >= np.min([n1,n2])/2.:
        lvl = np.int_(np.log2((n1-1)/(n-1.))+1)

    c = img
    ## wavelet set of coefficients.
    wave = np.zeros([lvl+1,n1,n2])
  
    for i in np.linspace(0,lvl-1,lvl):
        newh = np.zeros((1,n+(n-1)*(2**i-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ######Calculates c(j+1)
        ###### Line convolution
        if convol2d == 1:
            cnew = cp.convolve2d(c, H, mode='same', boundary='redlect')
        else:
            cnew = sc.convolve1d(c,newh[0,:],axis = 0, mode =mode)

            ###### Column convolution
            cnew = sc.convolve1d(cnew,newh[0,:],axis = 1, mode =mode)

 
      
        if newwave ==1:
            ###### hoh for g; Column convolution
            if convol2d == 1:
                hc = cp.convolve2d(cnew, H, mode='same', boundary='symm')
            else:
                hc = sc.convolve1d(cnew,newh[0,:],axis = 0, mode = mode)
 
                ###### hoh for g; Line convolution
                hc = sc.convolve1d(hc,newh[0,:],axis = 1, mode = mode)
            
            ###### wj+1 = cj-hcj+1
            wave[i,:,:] = c-hc
            
        else:
            ###### wj+1 = cj-cj+1
            wave[i,:,:] = c-cnew
 

        c = cnew
     
    wave[i+1,:,:] = c

    return wave

def iuwt(wave, convol2d =0):
    """
    Inverse Starlet transform.
    INPUTS:
        wave: wavelet decomposition of an image.
    OUTPUTS:
        out: image reconstructed from wavelet coefficients
    OPTIONS:
        convol2d:  if set, a 2D version of the filter is used (slower, default is 0)
        
    """
    mode = 'nearest'
    
    lvl,n1,n2 = np.shape(wave)
    h = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
    n = np.size(h)

    cJ = np.copy(wave[lvl-1,:,:])
    
    
    for i in np.linspace(1,lvl-1,lvl-1):
        
        newh = np.zeros((1,n+(n-1)*(2**(lvl-1-i)-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ###### Line convolution
        if convol2d == 1:
            cnew = cp.convolve2d(cJ, H, mode='same', boundary='symm')
        else:
          cnew = sc.convolve1d(cJ,newh[0,:],axis = 0, mode = mode)
            ###### Column convolution
          cnew = sc.convolve1d(cnew,newh[0,:],axis = 1, mode = mode)

        cJ = cnew+wave[lvl-1-i,:,:]

    out = np.reshape(cJ,(n1,n2))
    return out
    
