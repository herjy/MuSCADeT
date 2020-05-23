"""@package MuSCADeT

"""

import numpy as np

def asinh_norm(data, Q = 10, bands = [0,1,2], range = None):
    """Normalises frames in a data-cube for rgb display

    Parameter:
    ----------
    data: 'array'
        Cube of images with size nbxn1xn2
    Q: 'int'
        Stretching parameter for the arcsinh function.
    bands: 'array'
        An array of three values between 0 and nb-1 that contains the bands to use to display the rgb image
    range: 'array'
        if set to None, range will be taken as the min and max of the image.
    Returns:
    --------
    normimg: 'array'
        arcsinh-normalised array of slices of data.
    """
    img = data[bands]
    vmin = np.ma.min(img)
    if range is not None:
        range = np.ma.max(img)-vmin

    normimg = np.ma.array(np.arcsinh(Q * (img - vmin) / range) / Q)
    normimg/=np.max(normimg)

    normimg *= 255
    normimg[normimg<0] = 0
    normimg[normimg>255] = 255
    normimg = normimg.astype(np.uint8)
    normimg = np.transpose(normimg, axes=(1,2,0))
    return normimg