"""@package MuSCADeT


"""

import numpy as np
import matplotlib.pyplot as plt
import pca_ring_spectrum as pcas
import pyfits as pf
import wave_transform as mw
import pylab
import scipy.ndimage.filters as med


def mMCA(img, A,kmax, niter,mode = 'PCA', PCA = [2,10], harder = 0, pos = False,threshmode = 'mom',lvl = 6, soft = False, reweighting = 'none'):
    """
      mMCA runs the MuSCADeT algorithm over a cube of multi-band images.
  
      INPUTS:
          img: multiband cube with size nbxn1xn2 where nb is the number of bands and n1xn2,
the size of the images
          A: the mixing matrix. if mode is set to 'PCA', A will be ignored and can be set to 0
          kmax: detection threshold in units of noise standard deviation usually chosen between 3 and 5 
          niter: number of iterations of the MuSCADeT algorithm

      OUTPUTS:
          S: extracted sources
          A: mixing matrix, either given by the user or estimate by PCA with option mode ='PCA' 

      OPTIONS:
          mode: if set to 'PCA', the mixing matrix A will be estimated from PCA decomposition of the SEDs
          PCA: parameters for PCA sensitivity. if mode is set to 'PCA', the PCA estimator will take PCA[0]
as the number of sources to be extracted and PCA[1] as a sensitivity parameter to discriminate between
source. Values betwee 5 and 30 are usually recommended
          harder: if set to 1, 
          pos: if set to True, the output of the hard thresholding procedure is constrined to be positive
          threshmode: if set to 'mom', adaptive method of moments is used at every iteration to decrease the  threshold
          lvl: number of wavelet levels to use in the decompositions, default is 6.
          soft: if set to True, soft thresholding is used 

      EXAMPLE:

    
    """
    noisetab = np.array([ 0.8907963 ,  0.20066385,  0.08550751,  0.04121745,  0.02042497,
        0.01018976,  0.00504662,  0.00368314])
    n1,n2,nb = np.shape(img.T)
    
    if mode == 'PCA':
        Apca = PCA_initialise(img.T, PCA[0], angle = PCA[1])       
        Apca = np.multiply(Apca,[1./np.sum(Apca,0)])      
        A = Apca

    nb,ns = np.shape(A)
    X = np.zeros((ns,n1*n2))

    
    A = np.multiply(A,[1./np.sum(A,0)])
    

    

    AT = A.T

    

    [UA,EA, VA] = np.linalg.svd(A)
    EAmax = np.max(EA)
    mu1 = 2/linorm(A,10)
    mu = 2/EAmax
    
    mu = mu1

    Y = np.reshape(img,(nb,n1*n2))

    Ri = np.dot(AT,Y)
    sigma_y = np.zeros(nb)
    for i in np.linspace(0,nb-1,nb):
        sigma_y[i] = MAD(np.reshape(Y[i,:],(n1,n2)))*mu
        
    sigma1 = np.zeros(ns)
    sigma = sigma1+0
    for i in np.linspace(0,ns-1,ns):
        sigma1[i] = np.sqrt(np.sum( (AT[i,:]**2)*(sigma_y**2)))
        sigma[i]=MAD(np.reshape(Ri[i,:],(n1,n2)))*mu
 
    kmas = MOM(np.reshape(Ri,(ns,n1,n1)),sigma1,lvl)#15#np.max(np.dot(1/(mu*np.dot(AT,Y),1),mu*np.dot(AT,Y)))

    print(kmas)
    step = (kmas-kmax)/(niter-5)
    k = kmas

    per= np.zeros((ns,niter))
    w = np.zeros((ns,lvl,n1,n2))
    wmap = np.zeros((ns,lvl,n1,n2))
    S = np.zeros((ns,n1*n2))
    thmap = np.zeros((ns,lvl,n1,n2))
    ks = np.zeros(niter)
    sub = 0
    reweight = 0
    weight2 = 1

    for i in np.linspace(0,niter-1, niter):
            print(i)

            
            Sp = S

            
            R = mu*np.dot(AT, Y-np.dot(A,X))
            X = np.real(X+R)
            S = X
        
            wmax = np.zeros((ns))
            wm = np.zeros((ns,lvl))
            
            for j in np.linspace(0, ns-1, ns):

                w[j,:,:,:] = mw.wave_transform(np.reshape(S[j,:],(n1,n2)),lvl)
                for l in np.linspace(0,lvl-1,lvl):
                        wm[j,l] = np.max(np.abs(w[j,l,:,:]))/noisetab[l]
 #                       wmap[j,l,:,:] = wmap[j,l,:,:]/noisetab[l] 
                wmax[j] = np.max(wm[j,:])
                
                wmax[j] = wmax[j]/sigma[j]
                

            if threshmode == 'mom':
                    kmas = MOM(np.reshape(R,(ns,n1,n2)),sigma,lvl=lvl)
                    threshmom =np.max([kmas,kmax])
                    if threshmom <k:
                            k = threshmom
                            step = ((k-kmax)/(niter-i-6))
                            print('momy s threshold',threshmom)

            if reweighting != 'none':
                    for s in np.linspace(0,ns-1,ns):
                            thmap[s,:lvl-1,:,:] = (wmap[s-1,:lvl-1,:,:]) 
            
            for j in np.linspace(0, ns-1, ns):

                    kthr = np.max([kmax, k])
                        
                    Sj,wmap[j,:,:,:] = mr_filter(np.reshape(S[j,:],(n1,n2)),10,kthr,sigma[j],harder = harder, lvl = lvl,pos = pos,soft = soft)
                    S[j,:] = np.reshape(Sj,(n1*n2))           

          
            X=X
            ks[i] = kthr
            k = k-step
        
    S = np.zeros((ns,n1,n2))
    for l in np.linspace(0,ns-1,ns):
    
        S[l,:,:] = np.reshape((X[l,:]),(n1,n2)).T
    plt.plot(ks); plt.show()
    
    return S,A


def MOM(R,sigma,lvl = 6):
    """
    Estimates the best for a threshold from method of moments

      INPUTS:
          R: multi-sources cube with size nsxn1xn2 where ns is the number of sources
          and n1xn2, the size of an image
          sigma: noise standard deviation

      OUTPUTS:
          k: threshold level

      OPTIONS:
          lvl: number of wavelet levels used in the decomposition, default is 6.

      EXAMPLES
    """

    ns,n1,n2 = np.shape(R)
    
    noisetab = np.array([ 0.8907963 ,  0.20066385,  0.08550751,  0.04121745,  0.02042497,
        0.01018976,  0.00504662,  0.00368314])
    wmax = np.zeros((ns))
    wm = np.zeros((ns,lvl))
    w = np.zeros((ns,lvl,n1,n2))
    
    for j in np.linspace(0, ns-1, ns):
                w[j,:,:,:] = mw.wave_transform(R[j,:,:],lvl)
    for j in np.linspace(0, ns-1, ns):
                for l in np.linspace(0,lvl-2,lvl-1):
                        wm[j,l] = np.max(np.abs(w[j,l,:,:]))/noisetab[l]
                wmax[j] = np.max(wm[j,:])
                wmax[j] = wmax[j]/sigma[j]
                
    k = np.min(wmax)+(max(wmax)-min(wmax))/100
    return k

def MM(R,sigma,lvl = 6):
    n1,n2 = np.shape(R)
    
    noisetab = np.array([ 0.8907963 ,  0.20066385,  0.08550751,  0.04121745,  0.02042497,
                         0.01018976,  0.00504662,  0.00368314])
        
    wm = np.zeros((lvl))
    w = np.zeros((lvl,n1,n2))
                         
    w[:,:,:] = mw.wave_transform(R,lvl)
    for l in np.linspace(0,lvl-2,lvl-1):
        wm[l] = np.max(np.abs(w[l,:,:]))/noisetab[l]
    wmax = np.max(wm)/sigma

    k = (wmax)-(wmax)/100
    return k
    

def MAD(x):
    """
      Estimates noise level in an image from Median Absolute Deviation

      INPUTS:
          x: image 

      OUTPUTS:
          sigma: noise standard deviation

      EXAMPLES
    """
    meda = med.median_filter(x,size = (3,3))
    medfil = np.abs(x-meda)
    sh = np.shape(x)
    sigma = 1.48*np.median((medfil))
    return sigma

def mr_filter(img, niter, k, sigma,lvl = 6, pos = False, harder = 0,mulweight = 1, subweight = 0, addweight = 0, soft = False):
    """
      Computes wavelet iterative filtering on an image.

      INPUTS:
          img: image to be filtered
          niter: number of iterations (10 is usually recommended)
          k: threshold level in units of sigma
          sigma: noise standard deviation

      OUTPUTS:
          imnew: filtered image
          wmap: weight map

      OPTIONS:
          lvl: number of wavelet levels used in the decomposition, default is 6.
          pos: if set to True, positivity constrain is applied to the output image
          harder: if set to one, threshold levels are risen. This is used to compensate for correlated noise
          for instance
          mulweight: multiplicative weight (default is 1)
          subweight: weight map derived from other sources applied to diminish the impact of a given set of coefficient (default is 0)
          addweight: weight map used to enhance previously detected features in an iterative process (default is 0)
          soft: if set to True, soft thresholding is used
          
      EXAMPLES
    """


    levels = np.array([ 0.8907963 ,  0.20066385,  0.08550751,  0.04121745,  0.02042497,
        0.01018976,  0.00504662,  0.00368314])
    levels2g = np.array([ 0.94288346,  0.22998949,  0.10029194,  0.04860995,  0.02412084,
        0.01498695])

    shim = np.shape(img)
    n1 = shim[0]
    n2 = shim[1]
    M = np.zeros((lvl,n1,n2))
    M[:,:,:] = 0
    M[-1,:,:] = 1

    sh = np.shape(M)
    th = np.ones(sh)*(k)
    ##A garder
    th[0,:,:] = th[0,0,0]+1+5*harder
    th[1,:,:] = th[1,:,:]+5*harder
    th[2,:,:] = th[2,:,:]+5*harder
    th[3,:,:] = th[3,:,:]+2*harder
 #   th[4,:,:] = th[4,:,:]+5*harder
    
####################

    th =np.multiply(th.T,levels[:sh[0]]).T*sigma
    th[np.where(th<0)] = 0
    th[-1,:,:] = 0
    imnew = 0
    i =0

    R= img
    alpha = mw.wave_transform(R,lvl, newwave = 0)
    
    if pos == True :
         M[np.where(alpha-np.abs(addweight)+np.abs(subweight)-np.abs(th)*mulweight > 0)] = 1
    else:

         M[np.where(np.abs(alpha)-np.abs(addweight)+np.abs(subweight)-np.abs(th)*mulweight > 0)] = 1


    while i < niter:
        R = img-imnew
        
        alpha = mw.wave_transform(R,lvl,newwave = 1)

        if soft == True and i>0:
            alpha= np.sign(alpha)*(np.abs(alpha)-np.abs(addweight)+np.abs(subweight)-(th2g*mulweight))   

        Rnew = mw.iuwt(M*alpha)
        imnew = imnew+Rnew
        
        i = i+1
        
        
        imnew[np.where(imnew<0)]=0
        wmap = mw.wave_transform(imnew,lvl)
    return imnew,wmap


def linorm(A,nit):
    """
      Estimates the maximal eigen value of a matrix A

      INPUTS:
          A: matrix
          nit: number of iterations

      OUTPUTS:
          xn: maximal eigen value

       EXAMPLES

    """

    ns,nb = np.shape(A)
    x0 = np.random.rand(nb)
    x0 = x0/np.sqrt(np.sum(x0**2))

    
    for i in np.linspace(0,nit-1,nit):
        x = np.dot(A,x0)
        xn = np.sqrt(np.sum(x**2))
        xp = x/xn
        y = np.dot(A.T,xp)
        yn = np.sqrt(np.sum(y**2)) 
        if yn < np.dot(y.T,x0) :
            break
        x0 = y/yn

    return xn



def PCA_initialise(cube, ns, angle = 15,npca = 64):
    """
      Estimates the mixing matrix of of two sources in a multi band set of images

      INPUTS:
          cube: multi-band cube from which to extract mixing coefficients
          ns: number of mixed sources

      OUTPUTS:
          A0: mixing matrix

      OPTIONS:
          angle: sensitivity parameter. The angular resolution at which the algorithm has to look for PCA coefficients clustering
          npca: square root of the number of pixels to be used. Since too big images result in too big computation time
          we propose to downsample the image in order to get reasonable calculation time

      EXAMPLES
    """

    n,n,nband = np.shape(cube)
    cubep = cube+0.
    s = np.zeros(nband)
    for i in range(nband):
        s[i] = MAD(cube[:,:,i])
        cubep[:,:,i] = mr_filter(cube[:,:,i],10,3,s[i],harder = 0)[0]
    
    cubepca = np.zeros((np.min([n,npca]),np.min([n,npca]),nband))
    xk,yk = np.where(cubepca[:,:,0]==0)
    cubepca[xk ,yk,:] = cubep[xk*(n/npca),yk*(n/npca),:]
    lines = np.reshape(cubep,(n**2, nband))

   
    alphas, basis, sig= pcas.pca_ring_spectrum(cubepca[:,:,:].T,std = s)    
    ims0 = pcas.pca_lines(alphas,sig,angle, ns)

    vals = np.array(list(set(np.reshape(ims0,(npca*npca)))))

    vals = vals[np.where(vals>=0)]
    nsp = np.size(vals)
    
    spectras = np.ones([ns, nband])
    rank = nsp
    

    S_prior = np.zeros((n,n,np.size(vals)))
    xs,ys = np.where(S_prior[:,:,0]==0)
    count = 0

    for k in vals:
    
        x,y = np.where(ims0 == k)
        im = np.zeros((npca, npca))
        im[x,y] = 1

        S_prior[xs,ys,count] = im[np.int_(xs*(npca/n)), np.int_(ys*(npca/n))]#/(k+1)

        vecube = np.reshape(cubepca,(nband,npca*npca))

        ######Essai norm#####
        xcol,ycol=np.where(ims0==k)
        specs = np.reshape(cubepca[xcol,ycol,:],(len(xcol),nband))
        s1 =np.multiply(np.mean(specs,0),
                                      1/np.sum(np.reshape(cubepca,(npca**2,nband),0)))
        spectras[count,:]=s1/np.sum(s1,0)
        S_prior[:,:,count] = S_prior[:,:,count]*np.dot(cube,spectras[count,:])
        count = count+1
 
    S0 = np.reshape(S_prior[:,:,::-1],(ns,n*n))
    A0 = spectras.T
    
    return A0
