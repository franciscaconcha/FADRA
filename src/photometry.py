__author__ = 'fran'

from __future__ import division, print_function
import scipy as sp
import dataproc as dp
import copy

def centraldistances(data, c):
    """Computes distances for every matrix position from a central point c.
    :param data: array
    :type data: sp.ndarray
    :param c: center coordinates
    :type c: [float, float]
    :rtype: sp.ndarray
    """
    dy, dx = data.shape
    y, x = sp.mgrid[0:dy, 0:dx]
    return sp.sqrt((y - c[0]) * (y - c[0]) + (x - c[1]) * (x - c[1]))


def bipol(coef, x, y):
    """Polynomial fit for sky sustraction

    :param coef: sky fit polynomial coefficientes
    :type coef: sp.ndarray
    :param x: horizontal coordinates
    :type x: sp.ndarray
    :param y: vertical coordinates
    :type y: sp.ndarray
    :rtype: sp.ndarray
    """
    plane = sp.zeros(x.shape)
    deg = sp.sqrt(coef.size).astype(int)
    coef = coef.reshape((deg, deg))

    if deg * deg != coef.size:
        print("Malformed coefficient: " + str(coef.size) + "(size) != " + str(deg) + "(dim)^2")
    for i in sp.arange(coef.shape[0]):
        for j in sp.arange(i + 1):
            plane += coef[i, j] * (x ** j) * (y ** (i - j))

    return plane


def phot_error(phot, sky_std, n_pix_ap, n_pix_sky, gain, ron=None):
    """Calculates the photometry error

    :param phot: star flux
    :type phot: float
    :param sky: sky flux
    :type sky: float
    :param n_pix_ap: number of pixels in the aperture
    :type n_pix_ap: int
    :param n_pix_sky: number of pixels in the sky annulus
    :type n_pix_sky: int
    :param gain: gain
    :type gain: float
    :param ron: read-out-noise
    :type ron: float (default value: None)
    :rtype: float
    """

    # print("f,s,npa,nps,g,ron: %f,%f,%i,%i,%f,%f" %
    #       (phot, sky_std, n_pix_ap, n_pix_sky, gain, ron))

    if ron is None:
        print("Photometric error calculated without RON")
        ron = 0.0

    if gain is None:
        print("Photometric error calculated without Gain")
        gain = 1.0

    var_flux = phot/gain
    var_sky = sky_std**2 * n_pix_ap * (1 + float(n_pix_ap) / n_pix_sky)

    var_total = var_sky + var_flux + ron*ron*n_pix_ap

    return sp.sqrt(var_total)


    # N_e_star = gain * phot
    # N_e_sky = gain * sky
    # SNR = N_e_star / \
    #     sp.sqrt(N_e_star + n_pix_ap *
    #             (1.0 + float(n_pix_ap) / n_pix_sky) * (N_e_sky + ron))
    # return phot / SNR

#Defines subarray, size 2*rad+1, for centroid finding
def subarray(arr, y, x, rad):
    return arr[(y-rad):(y+rad+1), (x-rad):(x+rad+1)]


#Find centroid of small array
def centroid(arr):
    med = sp.median(arr)
    arr = arr-med
    arr *= arr > 0

    iy, ix = sp.mgrid[0:len(arr), 0:len(arr)]

    cy = sp.sum(iy*arr)/sp.sum(arr)
    cx = sp.sum(ix*arr)/sp.sum(arr)

    return cy, cx


def apphot(data, cs, sap, skydata, deg=1, gain=None, ron=None):
    """Do aperture photometry on data array

    :param data: image
    :type data: sp.ndarray
    :param cs: coordinates
    :type cs: [float, float]
    :param sap: apperture
    :type sap: float
    :param skydata: inner and outer radius for sky annulus or sky fit (coefficientes) and sky pixel mask
    :type skydata: [float, float] or [sp.ndarray, idx_array]
    :param deg: degree of the polynomial for the sky fit
    :type deg: int
    :param gain: gain
    :type gain: float
    :param ron: read-out-noise
    :type ron: float (default value: None)
    :rtype: [float, float, [coeff_list,list]]
    """

    d = centraldistances(data, cs)
    dy, dx = data.shape
    y, x = sp.mgrid[-cs[0]:dy - cs[0], -cs[1]:dx - cs[1]]

    # Compute sky correction
    # Case 1: skydata = [fit, map_of_sky_pixels]
    if isinstance(skydata[0], sp.ndarray):
        fit = skydata[0]
        idx = skydata[1]

    # Case 2: skydata = [inner_radius,outer_radius]
    else:
        import scipy.optimize as op
        idx = (d > skydata[0]) * (d < skydata[1])
        errfunc = lambda coef, x, y, z: (bipol(coef, x, y) - z).flatten()
        coef0 = sp.zeros((deg, deg))
        coef0[0, 0] = data[idx].mean()
        fit, cov, info, mesg, success = op.leastsq(errfunc, coef0.flatten(),
                                                   args=(x[idx], y[idx], data[idx]), full_output=1)

    # Apply sky correction
    n_pix_sky = idx.sum()
    sky = bipol(fit, x, y)
    sky_std = (data-sky)[idx].std()
    res = data - sky  # minus sky

    #compute the FWHM
    #Considering only those pixels with values higher than 3 times the sky noise and then only those within 2FWHM in addition
    # res2 = res.copy()
    # res2[res2<3*sky_std] = 0
    # fwhm = 2.355 * sp.sqrt((res2*d*d).sum()/len(res2[res2>0])/res2.sum())
    # res2[d>fwhm] = 0
    # fwhm = 2.355 * sp.sqrt((res2*d*d).sum()/len(res2[res2>0])/res2.sum())

    res2 = res[d < sap*4].ravel()
    d2 = d[d < sap*4].ravel()

    tofit = lambda d, h, sig: h*dp.gauss(d, sig, ndim=1)

    import scipy.optimize as op
    try:
        sig, cov = op.curve_fit(tofit, d2, res2,
                                sigma=1/sp.sqrt(sp.absolute(res2)),
                                p0=[max(res2), sap/3])
    except RuntimeError:
        sig = sp.array([0, 0, 0])

    fwhmg = 2.355*sig[1]

    #now photometry
    phot = float(res[d < sap].sum())

    #now the error
    if gain is None:
        error = None
    else:
        n_pix_ap = res[d < sap].sum()
        error = phot_error(phot, sky_std, n_pix_ap, n_pix_sky, gain, ron=ron)

    return phot, error, fwhmg, [fit, idx]
