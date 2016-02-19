"""@package MuSCADeT

"""

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
import matplotlib.colors as mc
import subprocess as sp
from MuSCADeT import MCA

def make_colour_sub(Sfile,Afile,Xfile,suffixe,prefix = './', cuts = ['0','0.1','-0.002','0.06','-0.002','0.03','0', '0.5']):
    """
     Creates colour images and visualisation of the residuals of the separation of estimated sources from a colour image.
     INPUTS:
        Sfile: name and path to a fits file with sources as estimated from MuSCADeT.
        Afile: name and path to a fits file with mixing coefficients as estimated from MuSCADeT.
        Xfile: name and path to a fits cube with original multi-band images used to feed MuSCADeT.
        suffixe: string that will be added at the end of the names of the png files showing the residuals.
    OUTPUTS:
        none. The code writes fits files and png files with the resulting residuals:
            prefix+'Colour_images.fits'
            prefix+'Red_residuals.fits'
            prefix+'Blue_residuals.fits'
            prefix+'Colour_residuals.fits'
            prefix+'S1_'+suffixe+'.png'
            prefix+'S2_'+suffixe+'.png'
            prefix+'Red_'+suffixe+'.png'
            prefix+'Blue_'+suffixe+'.png'
            prefix+'All_'+suffixe+'.png'
            prefix+'Res_'+suffixe+'.png'
    OPTIONS:
        prefix: string, location where to save fits and png files.
        cuts: colour cuts to apply to ds9 visualisation tool. cuts is an array
        with values [minR, maxR, minG, maxG, minB, maxB] where minR is the lower red cut and maxR
        is the maximum red cut (idem for Green and Blue)
    """
    
    S = pf.open(Sfile)[0].data
    A = pf.open(Afile)[0].data
    X = pf.open(Xfile)[0].data

    
    ns,nb = np.shape(A)
    nbb,n1,n2 = np.shape(X)

    sel = np.array([0,nbb/2,nbb-1])
 #   sel = np.array([0,3,5])
    if nbb != nb :
        A =A.T
    Subred = np.zeros((n1,n2,3)).T
    Subblue = np.zeros((n1,n2,3)).T
    XX=np.zeros((n1,n2,3)).T
    Res = Subblue+0
    for i in np.linspace(0,3-1,3):
        XX[i,:,:] = np.transpose(X[sel[i],:,:])
        Subred[i,:,:] = np.transpose(X[sel[i],:,:]-A[0,sel[i]]*S[0,:,:].T)
        Subblue[i,:,:] = np.transpose(X[sel[i],:,:]-A[1,sel[i]]*S[1,:,:].T)
        Res[i,:,:] = np.transpose(X[sel[i],:,:]-A[0,sel[i]]*S[0,:,:].T-A[1,sel[i]]*S[1,:,:].T)

    sigmar = MCA.MAD(XX[0,:,:])
    sigmag = MCA.MAD(XX[1,:,:])
    sigmab = MCA.MAD(XX[2,:,:])

    hdus = pf.PrimaryHDU(XX)
    lists = pf.HDUList([hdus])
    lists.writeto(prefix+'Colour_images.fits', clobber=True)
    hdus = pf.PrimaryHDU(Subred)
    lists = pf.HDUList([hdus])
    lists.writeto(prefix+'Red_residuals.fits', clobber=True)
    hdus = pf.PrimaryHDU(Subblue)
    lists = pf.HDUList([hdus])
    lists.writeto(prefix+'Blue_residuals.fits', clobber=True)

    hdus = pf.PrimaryHDU(Res)
    lists = pf.HDUList([hdus])
    lists.writeto(prefix+'Colour_residuals.fits', clobber=True)

    
    name_S1 = prefix+'S1_'+suffixe+'.png'
    name_S2 = prefix+'S2_'+suffixe+'.png'

##Cuts
    inf = cuts[6]
    maxi = cuts[7]

    sp.call('ds9 '+Sfile+' -scale limits '+inf+' '+maxi+' -cube 1 -zoom to fit -colorbar no -saveimage png '+name_S1,stdout=sp.PIPE, shell = True)
    sp.call('ds9 '+Sfile+' -scale limits '+inf+' '+maxi+' -cube 2 -zoom to fit -colorbar no -saveimage png '+name_S2,stdout=sp.PIPE, shell = True)

    ##Pour Refsdal
    ##    # Red levels
    infr = cuts[0]
    maxr = cuts[1]
    # Green levels
    infg = cuts[2]
    maxg = cuts[3]
# Blue levels
    infb = cuts[4]
    maxb = cuts[5]

    ##Pour Refsdal
    ##    # Red levels
    infrn = str(-5*sigmar)
    maxrn = str(5*sigmar)
# Blue levels
    infbn = str(-5*sigmab)
    maxbn = str(5*sigmab)
# Green levels

    infgn = str(-5*sigmag)
    maxgn = str(5*sigmag)

    name_red = prefix+'Red_'+suffixe+'.png'
    name_blue = prefix+'Blue_'+suffixe+'.png'
    name_colour =prefix+'All_'+suffixe+'.png'
    name_all = prefix+'Res_'+suffixe+'.png'
    
    sp.call('ds9 -rgbcube '+prefix+'Colour_images.fits -zoom to fit -colorbar no -rgb green -scale limits '+infg+' '+maxg+' -rgb red -scale limits '+infr+' '+maxr+' -rgb blue -scale limits '+infb+' '+maxb+' -saveimage png '+name_colour,stdout=sp.PIPE, shell = True)

    sp.call('ds9 -rgbcube '+prefix+'Red_residuals.fits -zoom to fit -colorbar no -rgb green -scale limits '+infg+' '+maxg+' -rgb red -scale limits '+infr+' '+maxr+' -rgb blue -scale limits '+infb+' '+maxb+' -saveimage png '+name_red,stdout=sp.PIPE, shell = True)
    sp.call('ds9 -rgbcube '+prefix+'Blue_residuals.fits -zoom to fit -colorbar no -rgb green -scale limits '+infg+' '+maxg+' -rgb red -scale limits '+infr+' '+maxr+' -rgb blue -scale limits '+infb+' '+maxb+' -saveimage png '+name_blue,stdout=sp.PIPE, shell = True)
    sp.call('ds9 -rgbcube '+prefix+'Colour_residuals.fits -zoom to fit -colorbar no -rgb green -scale limits '+infgn+' '+maxgn+' -rgb red -scale limits '+infrn+' '+maxrn+' -rgb blue -scale limits '+infbn+' '+maxbn+' -saveimage png '+name_all,stdout=sp.PIPE, shell = True)
    
    return 0
