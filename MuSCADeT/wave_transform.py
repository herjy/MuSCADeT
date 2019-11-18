"""@package MuSCADeT

"""

import numpy as np
import scipy.signal as cp
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sc

import MuSCADeT.wave_transform_base as wtb

try:
    import pysap
except ImportError:
    pysap_installed = False
else:
    pysap_installed = True
    
# TODO : terminate proper PySAP inegration (i.e. manage the 'pysap_transform' 
# object returned by wave_transform(), then pass it to iuwt())


def wave_transform(img, lvl, Filter='Bspline', newwave=1, convol2d=0, verbose=False):
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

    original_warning = "--> using original wavelet algorithm instead"
    
    if pysap_installed:
        if newwave == 0:
            coeffs, pysap_transform = wtb.uwt_pysap(img, lvl, Filter=Filter)
        else:
            if verbose:
                print("WARNING : PySAP does not support 2nd gen starlet")
                print(original_warning)
            coeffs = wtb.uwt_original(img, lvl, Filter='Bspline', 
                                     newwave=newwave, convol2d=convol2d)
            pysap_transform = None
    else:
        if verbose:
            print("WARNING : PySAP not installed or not found")
            print(original_warning)
        coeffs = wtb.uwt_original(img, lvl, Filter='Bspline', 
                                 newwave=newwave, convol2d=convol2d)
        pysap_transform = None
    return coeffs, pysap_transform


def iuwt(wave, newwave=1, convol2d=0, pysap_transform=None, verbose=False):
    """
    Inverse Starlet transform.
    INPUTS:
        wave: wavelet decomposition of an image.
    OUTPUTS:
        out: image reconstructed from wavelet coefficients
    OPTIONS:
        convol2d:  if set, a 2D version of the filter is used (slower, default is 0)
        
    """
    original_warning = "--> using original transform algorithm instead"
    
    if pysap_installed:
        if newwave == 0:
            if pysap_transform is None:
                raise RuntimeError("PySAP transform required for synthesis")
            recon = wtb.iuwt_pysap(wave, pysap_transform, fast=True)
        else:
            if verbose:
                print("WARNING : PySAP does not support 2nd gen starlet")
                print(original_warning)
            recon = wtb.iuwt_original(wave, convol2d=convol2d, newwave=newwave, fast=True)
    
    else:
        if verbose:
            print("WARNING : PySAP not installed or not found")
            print(original_warning)
        recon = wtb.iuwt_original(wave, convol2d=convol2d, newwave=newwave)
    return recon

    
