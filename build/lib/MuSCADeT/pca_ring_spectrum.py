"""@package MuSCADeT


"""

import numpy as np

import MuSCADeT.mk_pca as mk
from MuSCADeT import MCA

def pca_ring_spectrum(images, std = 0):
    """
    Decomposes a set of SEDs from multiband images into PCA and filters the less significant coefficients
    INPUTS:
        images: cube of muti-bandimages with size n1xn2xs where s is the number of bands and n1xn2, the size of each image
    OUTPUTS:
        alphas: PCA coefficients for each SED at each pixel location.
        basis: corresponding PCA basis.
        sig: noise as propagated into PCA space.


    EXAMPLE:
    """

    pad = 0
    images = images.T
    n1,n2,s = np.shape(images)
    res0 = images +0
    res = res0+0
    res1 = res+0.
    sigmamr = np.zeros(s)
    tr = res+0 #For thresholded images
    support = np.zeros((n1,n2))
    for j in range(s):
        sigmamr[j] = MCA.MAD(res0[:,:,j])
        res[:,:,j] = res1[:,:,j]
        x,y = np.where(res[:,:,j]==0)
        tr[x,y,j] = 0
        tr[:,:,-1] = 1


    support = np.prod(tr,2)

    support[np.where(support==0.0)] = 0
    support[np.where(support!=0.0)] = 1
    x00,y00 = np.where(support == 0)
    res[x00,y00,:] = 0

    x,y = np.where(support == 1)

    support1d = np.reshape(support,(n1*n2))
    x1d = np.where(support1d==1)

    spectrums = np.reshape(res[x,y,:],(np.size(x1d),s))
    alphas = np.zeros((np.size(x),n1*n2))
    alpha,base = mk.mk_pca(spectrums.T)

##Noise propagation in PCA space

    noise = np.multiply(np.random.randn(100,s),std.T)
    alphanoise = np.dot(base.T,noise.T)
    sig = np.zeros(2)
    sig[0] = np.std(alphanoise[0,:])
    sig[1] = np.std(alphanoise[1,:])


    count = 0
    for ind in np.reshape(x1d,np.size(x1d)):
        alphas[:,ind] = alpha[:,count]
        count = count+1


    return alphas, base, sig

def actg(X,Y):
    """
    Computes the arctan(x/y) of two vectors.
    INPUTS:
        X: 1-d vector
        Y: 1-d vector
    OUTPUTS:
        angle: 1-d vector with the result of arctan(X/Y)

    EXAMPLE:
    """

    if X >0 and Y>=0:
        angle = np.arctan(Y/X)
    if X >0 and Y<0:
        angle = np.arctan(Y/X)+2*np.pi
    if X<0:
        angle = np.arctan(Y/X)+np.pi
    if X ==0 and Y>0:
        angle = np.pi/2
    if X ==0 and Y<=0:
        angle = 3*np.pi/2
    return angle


def pca_lines(alphas, sig, dt, ns, alpha0 = [0,40], plot = 0):
    """
    Finds alignments in PCA coefficients and identifies corresponding structures in direct space. It is actually a simple angular clustering algorithm.
    INPUTS:
        alphas: PCA coefficients.
        sig: noise levels in the two first PCA components
        dt: angular resolution at which the algorithm has to discriminate between coefficients of a same group
        ns: number of alignments to identify.
    OUTPUTS:
        images: 2-d map of strucutres with same colours. Each structure has all its pixels set to the same value.
        Pixels identified as non-significant are set to 0.
    EXAMPLE:
    """
    dt = dt*np.pi/180
    n1,n2 = np.shape(alphas)

    #coefficients dans le bruit
    noisy = np.zeros(n2)
    noisy[np.where(np.abs(alphas[0,:])<5*sig[0])] = 1
    noisy[np.where(np.abs(alphas[1,:])<5*sig[1])] = noisy[np.where(np.abs(alphas[1,:])<5*sig[1])] +1
    alphas[:,np.where(noisy==2)] = 0.

    #norm
    norm = (alphas[0,:]**2 + alphas[1,:]**2)



    alphas[:,np.where(norm == 0)] = 0
    #Rescaling des angles
    alphas[0,:] = np.sign(alphas[0,:])*np.abs(alphas[0,:]/np.max(np.abs(alphas[0,:])))
    alphas[1,:] = np.sign(alphas[1,:])*np.abs(alphas[1,:]/np.max(np.abs(alphas[1,:])))

    X = alphas[0,:]
    Y = alphas[1,:]
    angle = np.zeros(np.size(X))
    for i in range(np.size(angle)):
        angle[i] = actg(X[i],Y[i])
    angle[np.where( norm==0)] = 0


    #Angles a zero non pris en compte
    loc = np.where(angle!=0)
    theta = angle[loc]

    normtrunc = norm[loc]
    cluster = np.zeros(np.size(theta))*2
    attractors = np.zeros(ns)
    attractors[0] = np.pi/2.
    for h in range(ns-1):
        attractors[h+1] = attractors[h]+ 2*np.pi/(ns)
    find = 0
    last = 0

    beta = np.zeros(2*int(np.pi/dt))
    count = 0
    maxi = np.zeros(ns)
    loctheta = np.zeros(2*int(np.pi/dt))
    k = 0

    if np.sum(alpha0)!=0:
        attractors = np.array(alpha0)*np.pi/180.
    else:
        while 1:
            isdone = 0
            count = 0
            #On parcours les angles pour leur attribuer chacun un attracteur
            for T in theta:
                #Distance angle attracteur
                dist = np.abs(T-attractors)

                #Correction du passage 2pi-0
                bigloc =np.where(dist>=np.pi)
                if np.size(bigloc)>0:
                    dist[bigloc] = 2*np.pi-dist[bigloc]
                find = np.where(dist == np.min(dist))[0]
                #Attribution de l'attracteur
                cluster[count]=find


                count = count+1

            if last ==1:
                break
            #Recomputing attractors by averaging over the detected angles
            oldattractors = attractors+0.
            for j in range(ns):
                sample = theta[np.where(cluster == j)]
                if np.size(sample) ==0:
                    attractors[j] = oldattractors[j]+np.pi/2.
                else:
                    if np.max(sample)-np.min(sample) >= np.pi:
                        sample[np.where(sample<np.pi)] = sample[np.where(sample<np.pi)] + 2*np.pi

                    if np.mean(sample) >2*np.pi:
                        attractors[j] = np.median(sample)-2*np.pi
                    else:
                        attractors[j] = np.median(sample)
                    if attractors[j] == oldattractors[j]:
                        isdone = isdone+1
            if isdone == ns:
                last = 1
    ###

    #Select only the coefficients in an given angular proximity
    locky = np.zeros(np.size(theta))-1.
    for i in range(ns):
        distance = np.abs(theta-attractors[i])

        bigloc = np.where(distance >= np.pi)
        if np.size(bigloc) >0:
            distance[bigloc] = 2*np.pi-distance[bigloc]


        distance[np.where(theta == 0)] = 0
        locky[np.where(distance<dt/2)]=i
        bigloc = np.where(distance >= np.pi)
        if np.size(bigloc) >0:
            distance[bigloc] = 2*np.pi-distance[bigloc]


        distance[np.where(theta == 0)] = 0
        locky[np.where(distance<dt/2)]=i
    ###############

    theta[np.where(locky ==-1)]=0

    locator = np.zeros(np.size(angle))-1.
    locator[loc] = locky

#### To comment or not to comment ####
#    plt.plot(alphas[0,:],alphas[1,:],'x')
#    plt.plot([0,np.cos(attractors[0])],[0,np.sin(attractors[0])])
#    plt.plot([0,np.cos(attractors[1])],[0,np.sin(attractors[1])])
#    plt.show()

    images = np.zeros((int(n2**0.5), int(n2**0.5)))
    images[:,:]=-1
    x,y = np.where(np.zeros((int(n2**0.5), int(n2**0.5)))==0)


    clus = angle +0.
    for j in range(np.size(angle)):

 #       clus[j] = np.where(mini==j)#np.where(np.abs(axis-angle[j]) == np.min(np.abs(axis-angle[j])))[0]+1
 #       if norm0[j] == 0:
 #               clus[j] =-2
        images[x[j], y[j]] = locator[j]

    n_clus = ns+1

    colors = [[0.6,0,0],np.array([135, 233, 144])/255.,[0,0,0]]#plt.cm.Spectral(np.linspace(0, 1, n_clus))

    for k, col in zip(set(locator), colors):
        if k == -1:
            # Black used for noise.
            col = [0,0,0.7]

## #       class_member_mask = (clus== k)
        xy = alphas[0:2,np.where(locator == k)[0]]
        if plot == True:
            plt.figure(1)
            plt.plot(xy[0,:], xy[1,:], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')


##   #     xk.XKCDify(ax, expand_axes=True)
 #   plt.axis('equ)
    if plot == True:
        plt.figure(20)
        plt.imshow(np.flipud(images), interpolation ='nearest')#; plt.colorbar()
        plt.axis('off')
        plt.show()

    return images










