import numpy as np
import scipy.signal as scs
import scipy.ndimage.filters as scf


def uwt_pysap(img, lvl, Filter='Bspline', n_omp_threads=0):
    """private function : Wavelet transform through PySAP"""

    import pysap
    
    lvl -= 1  # TODO : should be cleaned

    def pysap2muscadet(a_list):
        return np.asarray(a_list)

    nb_scale = lvl+1  # + 1 for the coarsest scale

    if Filter == 'Bspline':  # = 1st starlet (2nd gen not yet implemented in PySAP)

        transform_name = 'BsplineWaveletTransformATrousAlgorithm'

        # note that if 'n_omp_threads' is not provided, 
        # PySAP will automatically set it the 
        # max number of CPUs available minus 1

        transform_obj = pysap.load_transform(transform_name)
        transform = transform_obj(nb_scale=nb_scale, verbose=1, 
                                  padding_mode='symmetric',
                                  nb_procs=n_omp_threads)

    else:
        raise NotImplementedError("Only sarlet transform is supported for now")

    # set the image
    transform.data = img
    transform.analysis()
    coeffs = transform.analysis_data
    return pysap2muscadet(coeffs), transform


def iuwt_pysap(wave, transform, fast=True):
    """private function : Inverse wavelet transform through PySAP"""

    import pysap
    
    def muscadet2pysap(a):
        a_list = []
        for i in range(a.shape[0]):
            a_list.append(a[i, :, :])
        return a_list

    if fast:
        # for 1st gen starlet the reconstruction can be performed by summing all scales 
        recon = np.sum(wave, axis=0)

    else:
        # use set the analysis coefficients
        transform.analysis_data = muscadet2pysap(wave)
        image = transform.synthesis()
        recon = image.data

    return recon


def uwt_original(img, lvl, Filter = 'Bspline', newwave = 1, convol2d = 0):
    """private function : Wavelet transform through original MuSCADeT algorithm"""

    mode = 'nearest'
    
    lvl = lvl-1  # TODO : should be cleaned
    
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
        h = [np.float(1./16.), np.float(1./4.), np.float(3./8.), np.float(1./4.), np.float(1./16.)]
    else:
        h = [1./4,1./2,1./4]
    n = np.size(h)
    h = np.array(h)
    
    # if n+2.**(lvl-1)*(n-1) >= np.min([n1,n2])/2.:
    #     lvl = int(np.log2((n1-1.)/(n-1.))+1.)

    c = img
    ## wavelet set of coefficients.
    wave = np.zeros([lvl+1,n1,n2])
  
    for i in range(lvl):
        newh = np.zeros((1,n+(n-1)*(2**i-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ######Calculates c(j+1)
        ###### Line convolution
        if convol2d == 1:
            cnew = scs.convolve2d(c, H, mode='same', boundary='symm')
        else:
            cnew = scf.convolve1d(c,newh[0,:],axis = 0, mode =mode)

            ###### Column convolution
            cnew = scf.convolve1d(cnew,newh[0,:],axis = 1, mode =mode)

 
      
        if newwave ==1:
            ###### hoh for g; Column convolution
            if convol2d == 1:
                hc = scs.convolve2d(cnew, H, mode='same', boundary='symm')
            else:
                hc = scf.convolve1d(cnew,newh[0,:],axis = 0, mode = mode)
 
                ###### hoh for g; Line convolution
                hc = scf.convolve1d(hc,newh[0,:],axis = 1, mode = mode)
            
            ###### wj+1 = cj-hcj+1
            wave[i,:,:] = c-hc
            
        else:
            ###### wj+1 = cj-cj+1
            wave[i,:,:] = c-cnew
 

        c = cnew
     
    wave[i+1,:,:] = c

    return wave

def iuwt_original(wave, convol2d =0, newwave=1, fast=True):
    """private function : Inverse wavelet transform through original MuSCADeT algorithm"""
    if newwave == 0 and fast:
        # simply sum all scales, including the coarsest one
        return np.sum(wave, axis=0)

    mode = 'nearest'
    
    lvl,n1,n2 = np.shape(wave)
    h = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
    n = np.size(h)

    cJ = np.copy(wave[lvl-1,:,:])
    
    
    for i in range(1, lvl):
        
        newh = np.zeros( ( 1, int(n+(n-1)*(2**(lvl-1-i)-1)) ) )
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ###### Line convolution
        if convol2d == 1:
            cnew = scs.convolve2d(cJ, H, mode='same', boundary='symm')
        else:
          cnew = scf.convolve1d(cJ,newh[0,:],axis = 0, mode = mode)
            ###### Column convolution
          cnew = scf.convolve1d(cnew,newh[0,:],axis = 1, mode = mode)

        cJ = cnew+wave[lvl-1-i,:,:]

    out = np.reshape(cJ,(n1,n2))
    return out

