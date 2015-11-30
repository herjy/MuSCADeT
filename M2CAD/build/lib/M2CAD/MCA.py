import numpy as np
import matplotlib.pyplot as plt
import pca_ring_spectrum as pcas
import pyfits as pf
import wave_transform as mw
import pylab
import scipy.ndimage.filters as med

            

def mMCA(img, A,kmax, niter, X0 = 0, PCA = [0,0,10], penalty = 0, harder = 0, reweighting = 'none', wmode='add', grad = True, jl = 0,pos = False,threshmode = 'mom',mode = 'PCA',lvl = 6, chi2 = 1,soft = False,disjoint = False):
    """
    Solves argmin(||y - Ax||^2 + ||dec(x)||
    INPUT:
        img:    multi-band data cube of size n1,n2,n where n is the number of bands observed and n1*n2,
                   the number of pixels per band.
        A:       Mixing matrix containing the spectral energy distribution of the expected sources in img
        dec:   Operator that decomposes sources in an adequate dictionnary for sparsity
        undec:Inverse operation of dec
        kmax: Minimum detection level (usually between 3 and 5 sigma)
        niter:   Number of iterations
        
    OPTIONS:
        X0:     Prior on the sources provided as an array with shape n1*n2,ns with ns, the expected number of sources
        PCA:  Estimates the mixing matrix from PCA decomposition of the SED in each pixel (requires mode = 'PCA').
                  Should be provided as an array: [ns,angle,std], where ns is the expected number of sources,
                  angle is the PCA angular resolution (usually between 5 and 30) and std the detection level in PCA coefficients.
        Threshmode: if 'minmax', the minimum of maximum policy is applied to set the thresholds
        mode: if 'PCA', estimates A from PCA.
        comax: Maximum number of coefficients to use in spectra reconstruction
        reweighting: string that specifies the type of reweighting. Can be 'none', 'pen, 'sub', 'frac' or 'all'.
    """
    noisetab = np.array([ 0.8907963 ,  0.20066385,  0.08550751,  0.04121745,  0.02042497,
        0.01018976,  0.00504662,  0.00368314])
    n1,n2,nb = np.shape(img.T)
    
    if mode == 'PCA':
        [X,Apca] = PCA_initialise(img.T, PCA[0], angle = PCA[1], mode = 'direct')
        
        Apca = np.multiply(Apca,[1./np.sum(Apca,0)])
        
        plt.plot(Apca); plt.show()
        
        A = Apca
    else:
        nbb,ns = np.shape(A)
        if np.sum(X0) == 0:
            X = np.zeros((ns,n1*n2))
        else:
            X = np.reshape(X0,(ns,np.size(X0[0,:,:])))
    
    A = np.multiply(A,[1./np.sum(A,0)])
    nb,ns = np.shape(A)

    
    if grad == False:
        AT = np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)/np.sum(np.linalg.inv(np.dot(A.T,A)))
    else:
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
            if grad == False:
                R = np.dot(AT,Y)
                S=R
            else:
            
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
####################test
                    if 1:
                        
                                          
                    
                        if reweighting== 'pen':#'pen sub':# or 'pen frac' or 'all':
                            print('zizi')
                            reweight = (thmap[j,:,:,:])*sigma[j]/sigma[j-1]#/sigma[j-1])
                        if reweighting == 'sub':# or 'pen sub': #or 'sub frac' or 'all':
                       
                            sub = (thmap[j-1,:,:,:])*sigma[j-1]/sigma[j]#/sigma[j])
                        if reweighting == 'frac':# or 'pen frac' or 'sub frac' or 'all':

                            weight2 = 1/(np.abs(thmap[j-1,:,:,:]/sigma[j])+0.00001)#
                        if reweighting == 'pen sub':
                        
                              sub = (thmap[j-1,:,:,:])*sigma[j-1]/sigma[j]
                              reweight = (thmap[j,:,:,:])
                        if reweighting == 'sub frac':
                            sub = (thmap[j-1,:,:,:])*sigma[j-1]/sigma[j]
                            weight2 = 1/(np.abs(thmap[j-1,:,:,:]/sigma[j])+0.00001)

                        if reweighting == 'pen frac':
                            reweight = (thmap[j,:,:,:])*sigma[j]/sigma[j-1]
                            weight2 = 1/(np.abs(thmap[j-1,:,:,:]/sigma[j])+0.00001)

                    kthr = np.max([kmax, k])
                        
                    if disjoint == True:
                        if i > 9*niter/10:
                            dis = wmap[j-1,:,:,:]
                        else:
                            dis = 0
                        Sj,wmap[j,:,:,:] = mr_filter(np.reshape(S[j,:],(n1,n2)),10,kthr,sigma[j],harder = harder,subweight = sub, mulweight = weight2,addweight = reweight,disjoint = dis, lvl = lvl,pos = pos,soft = soft)
                        S[j,:] = np.reshape(Sj,(n1*n2))
                    else:
                        Sj,wmap[j,:,:,:] = mr_filter(np.reshape(S[j,:],(n1,n2)),10,kthr,sigma[j],harder = harder, subweight = sub, mulweight = weight2,addweight = reweight, weightmode = wmode, lvl = lvl,pos = pos,soft = soft)
                        S[j,:] = np.reshape(Sj,(n1*n2))

            ####Test estimation de A
 #           if i >niter/2:
#                Anew = np.dot(Y,np.dot(S.T,np.linalg.inv(np.dot(S,S.T))))
#                Anew = np.multiply(Anew,[1./np.sum(Anew,0)])
#                A = (A+Anew)/2
            

            if grad == False:
                X=X+S
            else:
                X=X
            ks[i] = kthr
            k = k-step
        
    S = np.zeros((ns,n1,n2))
    for l in np.linspace(0,ns-1,ns):
    
        S[l,:,:] = np.reshape((X[l,:]),(n1,n2)).T
    plt.plot(ks); plt.show()
    if chi2 == True:
        sig = np.zeros((nb))
        serr = sig+0
        for i in np.linspace(0,nb-1,nb):
            err = np.sum((img[i,:,:]-A[i,0]*S[0,:,:]-A[i,1]*S[1,:,:]))#**2/(n1*n2)
            serr[i] = (err)**2
        Chi = np.sum(serr/sigma_y**2)/nb
        print(Chi)
        
    else:
        Chi = 0
    return S,A,Chi


def MOM(R,sigma,lvl = 6):
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
    

def MAD(x):
        meda = med.median_filter(x,size = (3,3))
        medfil = np.abs(x-meda)
        sh = np.shape(x)
        sigma = 1.48*np.median((medfil))
        return sigma

def mr_filter(img, niter, k, sigma,lvl = 6, pos = False, harder = 0,mulweight = 1, subweight = 0, disjoint = [0], addweight = 0, weightmode = 'add', soft = False):

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
    th[0,:,:] = th[0,0,0]+1+10*harder
    th[1,:,:] = th[1,:,:]+15*harder
    th[2,:,:] = th[2,:,:]+10*harder
 #   th[3,:,:] = th[3,:,:]+2*harder
 #   th[4,:,:] = th[4,:,:]+5*harder
    
####################

 #   th2g = np.ones(sh)

    th =np.multiply(th.T,levels[:sh[0]]).T*sigma
#    th2g = np.multiply(th.T,levels2g[:sh[0]]).T*sigma*(k)
    
    th[np.where(th<0)] = 0

    th[-1,:,:] = 0

 #   th2g[np.where(th<0)] = 0
 #   th2g[0,:,:] = th[0,0,0]+1
 #   th2g[-1,:,:] = 0

    imnew = 0
    i =0

    R= img
    alpha = mw.wave_transform(R,lvl, newwave = 0)
    
    if pos == True :
         M[np.where(alpha-np.abs(addweight)+np.abs(subweight)-np.abs(th)*mulweight > 0)] = 1
    else:
 #        plt.plot(alpha[3,:,:],'r', addweight[3,:,:],'b')
         M[np.where(np.abs(alpha)-np.abs(addweight)+np.abs(subweight)-np.abs(th)*mulweight > 0)] = 1


    while i < niter:
        R = img-imnew
        
        alpha = mw.wave_transform(R,lvl,newwave = 1)

        if soft == True and i>0:
            alpha= np.sign(alpha)*(np.abs(alpha)-np.abs(addweight)+np.abs(subweight)-(th2g*mulweight))   
 #       M[np.where(np.abs(alpha)-th > 0)] = 1
 #      alpha[np.where(disjoint!=0)]=0
        
        Rnew = mw.iuwt(M*alpha)
        imnew = imnew+Rnew
        
        i = i+1
        
        
        imnew[np.where(imnew<0)]=0
        wmap = mw.wave_transform(imnew,lvl)
    return imnew,wmap


def linorm(A,nit):
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



def PCA_initialise(cube, ns, angle = 15, mode = 'PCA',comax = 2, npca = 64):
    #INPUTS:
    #   cube: data cube of size NxNxNband with NxN the number of pixels in
    #             each band and Nband, te number of bands in the cube.
    #   ns: number of expected sources
    #   angle: angle resolution for PCA discrimination
    #   std: PCA component threshold
    #   comax: Maximum number of coefficients to use in spectra reconstruction
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
        if mode == 'PCA':
            for k in vals:
                imsline = np.reshape(ims0,np.size(ims0))
                x = np.where(imsline == k)
                x = np.reshape(x,(np.size(x)))
                alphak = alphas[:comax,x]
                basisk = basis[:,:comax]

                
                specsk = np.dot(basisk,alphak)
                si = np.mean(specsk,1)
                
                spectras[count,:] = si/np.sum(si)
                count = count+1
       
        if mode == 'direct':
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
        
        return S0,A0
