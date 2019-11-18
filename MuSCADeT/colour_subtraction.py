"""@package MuSCADeT

"""

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
#from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
import matplotlib.colors as mc
import subprocess as sp
import scipy.signal as sc
import os

from MuSCADeT import MCA

def make_colour_sub(Sfile,Afile,Xfile,suffixe,prefix = './', cuts = ['0','0.1','-0.002','0.06','-0.002','0.03','0', '0.5'], display=True, simu_folder='', sel=[0,0,0], PSF=0):
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
        display : display of the images 
        simu_folder : folder path to save the images 
        sel : vector containing the index of the band to select for the RGB images 
        PSF : data cube containgin the PSF in the different bands
    """
    
    S = pf.open(Sfile)[0].data
    A = pf.open(Afile)[0].data
    X = pf.open(Xfile)[0].data
    AX1 = np.zeros(np.shape(X[0]))
    AX2 = np.zeros(np.shape(X[1]))
   
    ns,nb = np.shape(A)
    nbb,n1,n2 = np.shape(X)
    
    if np.sum(sel)==0:
        sel = np.array([0,nbb/2,nbb-1])

    if nbb != nb :
        A =A.T
    Subred = np.zeros((n1,n2,nbb)).T
    Subblue = np.zeros((n1,n2,nbb)).T
    XX=np.zeros((n1,n2,nbb)).T
    Res = Subblue+0

    Subred_disp = np.zeros((n1,n2,3)).T
    Subblue_disp = np.zeros((n1,n2,3)).T
    XX_disp = np.zeros((n1,n2,3)).T
    Res_disp = Subblue+0
    for i in range(nbb):
            if np.sum(PSF) != 0: 
                    AX1 = sc.fftconvolve( A[0,i]*S[0,:,:].T,PSF[i],mode='same')
                    AX2 = sc.fftconvolve( A[1,i]*S[1,:,:].T,PSF[i],mode='same')
                    if i<3:
                        AX1_disp = sc.fftconvolve( A[0,sel[i]]*S[0,:,:].T,PSF[sel[i]],mode='same')
                        AX2_disp = sc.fftconvolve( A[1,sel[i]]*S[1,:,:].T,PSF[sel[i]],mode='same')
            else:
                    AX1 = A[0,i]*S[0,:,:].T
                    AX2 = A[1,i]*S[1,:,:].T
                    if i<3:
                        AX1_disp = A[0,sel[i]]*S[0,:,:].T
                        AX2_disp = A[1,sel[i]]*S[1,:,:].T
            XX[i,:,:] = (X[i,:,:])
            Subred[i,:,:] = (X[i,:,:]-AX1)
            Subblue[i,:,:] = (X[i,:,:]-AX2)
            Res[i,:,:] = (X[i,:,:]-AX1-AX2)

            if i<3:
                XX_disp[i,:,:] = (X[sel[i],:,:])
                Subred_disp[i,:,:] = (X[sel[i],:,:]-AX1_disp)
                Subblue_disp[i,:,:] = (X[sel[i],:,:]-AX2_disp)
                Res_disp[i,:,:] = (X[sel[i],:,:]-AX1_disp-AX2_disp)

    sigmar = MCA.MAD(XX[sel[0],:,:])
    sigmag = MCA.MAD(XX[sel[1],:,:])
    sigmab = MCA.MAD(XX[sel[2],:,:])

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

###For display
    hdus = pf.PrimaryHDU(XX_disp)
    lists = pf.HDUList([hdus])
    lists.writeto(prefix+'Colour_images_disp.fits', clobber=True)
    hdus = pf.PrimaryHDU(Subred_disp)
    lists = pf.HDUList([hdus])
    lists.writeto(prefix+'Red_residuals_disp.fits', clobber=True)
    hdus = pf.PrimaryHDU(Subblue_disp)
    lists = pf.HDUList([hdus])
    lists.writeto(prefix+'Blue_residuals_disp.fits', clobber=True)
    hdus = pf.PrimaryHDU(Res_disp)
    lists = pf.HDUList([hdus])
    lists.writeto(prefix+'Colour_residuals_disp.fits', clobber=True)


    name_S1 = prefix+'S1_'+suffixe+'.png'
    name_S2 = prefix+'S2_'+suffixe+'.png'

##Cuts
    inf = cuts[6]
    maxi = cuts[7]
    
    if display:
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
    
    if display:
            sp.call('ds9 -rgbcube '+prefix+'Colour_images_disp.fits -zoom to fit -colorbar no -rgb green -scale limits '+infg+' '+maxg+' -rgb red -scale limits '+infr+' '+maxr+' -rgb blue -scale limits '+infb+' '+maxb+' -saveimage png '+name_colour,stdout=sp.PIPE, shell = True)
            sp.call('ds9 -rgbcube '+prefix+'Red_residuals_disp.fits -zoom to fit -colorbar no -rgb green -scale limits '+infg+' '+maxg+' -rgb red -scale limits '+infr+' '+maxr+' -rgb blue -scale limits '+infb+' '+maxb+' -saveimage png '+name_red,stdout=sp.PIPE, shell = True)
            sp.call('ds9 -rgbcube '+prefix+'Blue_residuals_disp.fits -zoom to fit -colorbar no -rgb green -scale limits '+infg+' '+maxg+' -rgb red -scale limits '+infr+' '+maxr+' -rgb blue -scale limits '+infb+' '+maxb+' -saveimage png '+name_blue,stdout=sp.PIPE, shell = True)
            sp.call('ds9 -rgbcube '+prefix+'Colour_residuals_disp.fits -zoom to fit -colorbar no -rgb green -scale limits '+infgn+' '+maxgn+' -rgb red -scale limits '+infrn+' '+maxrn+' -rgb blue -scale limits '+infbn+' '+maxbn+' -saveimage png '+name_all,stdout=sp.PIPE, shell = True)

    os.remove(prefix+'Colour_images_disp.fits')
    os.remove(prefix+'Red_residuals_disp.fits')
    os.remove(prefix+'Blue_residuals_disp.fits')
    os.remove(prefix+'Colour_residuals_disp.fits')
    return 0
