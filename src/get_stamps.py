import dataproc as dp
import copy
import scipy as sp

def zscale(img,  trim = 0.05, contr=1, mask=None):
    """Returns lower and upper limits found by zscale algorithm for improved contrast in astronomical images.

:param mask: bool ndarray
    True are good pixels, pixels marked as False are ignored
:rtype: (min, max)
    Minimum and maximum values recommended by zscale
"""
    if not isinstance(img, sp.ndarray):
        img = sp.array(img)
    if mask is None:
        mask = (sp.isnan(img) == False)

    itrim = int(img.size*trim)
    x = sp.arange(mask.sum()-2*itrim)+itrim

    sy = sp.sort(img[mask].flatten())[itrim:img[mask].size-itrim]
    a, b = sp.polyfit(x, sy, 1)

    return b, a*img.size/contr+b

def centroid(orig_arr, medsub=True):
    """Find centroid of small array
    :param arr: array
    :type arr: array
    :rtype: [float,float]
    """
    arr = copy.copy(orig_arr)

    if medsub:
        med = sp.median(arr)
        arr = arr - med

    arr = arr * (arr > 0)
    print(arr)

    iy, ix = sp.mgrid[0:len(arr), 0:len(arr)]

    cy = sp.sum(iy * arr) / sp.sum(arr)
    cx = sp.sum(ix * arr) / sp.sum(arr)

    return cy, cx


def get_stamps(sci, mdark, mflat, target_coords, stamp_rad):
    """

    :param sci:
    :type sci: AstroDir
    :param target_coords: [[t1x, t1y], [t2x, t2y], ...]
    :param stamp_rad:
    :return:
    """
    all_cubes, dark_cubes, flat_cubes = [], [], []
    data = sci.readdata()

    for c in target_coords:
        cx, cy = c[0], c[1]
        cube, dcube, fcube = [], [], []
        stamp = data[0][(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)]
        cx0, cy0 = centroid(stamp)
        cx = cx - stamp_rad + cx0.round()
        cy = cy - stamp_rad + cy0.round()
        stamp = data[0][(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)]
        cube.append(stamp)
        dark_stamp = mdark[(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)]
        flat_stamp = mflat[(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)]
        dcube.append(dark_stamp)
        fcube.append(flat_stamp)
        for d in data[1:]:
            stamp = d[(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)]
            cx0, cy0 = centroid(stamp)
            cx = cx - stamp_rad + cx0.round()
            cy = cy - stamp_rad + cy0.round()
            cube.append(stamp)
            dark_stamp = mdark[(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)]
            flat_stamp = mflat[(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)]
            dcube.append(dark_stamp)
            fcube.append(flat_stamp)
        all_cubes.append(cube)
        dark_cubes.append(dcube)
        flat_cubes.append(fcube)

    return all_cubes, dark_cubes, flat_cubes

def CPUphot(sci, dark, flat, sky):
    n_targets = len(sci)
    for n in range(n_targets):
        for s, d, f in zip(sci[n], dark[n], flat[n]):
            red = (s - d)/(f - d)



def photometry(sci, mbias, mdark, mflat, target_coords, stamp_rad, sky, gpu=False):
    sci_stamps, dark_stamps, flat_stamps = get_stamps(sci, mdark, mflat, target_coords, stamp_rad)

    if gpu:
        return GPUphot(sci_stamps, dark_stamps, flat_stamps, sky)
    else:
        return CPUphot(sci_stamps, dark_stamps, flat_stamps, sky)



io = dp.AstroDir("/media/Fran/2011_rem/rawsci70/raw6")
# OJO coordenadas van Y,X
res = get_stamps(io, None, None, None, [[577, 185], [488, 739]], 20)
print(len(res), len(res[0]), res[0][0].shape)

import matplotlib.pyplot as plt

fig1 = plt.figure()
l = len(res[0])
n = len(res)

for i in range(1, l + 1):
    plt.subplot(n, l, i)
    lmin, lmax = zscale(res[0][i - 1])
    plt.imshow(res[0][i - 1], vmin=lmin, vmax=lmax, cmap=plt.get_cmap('gray'))
    plt.subplot(n, l, l + i)
    lmin, lmax = zscale(res[1][i - 1])
    plt.imshow(res[1][i - 1], vmin=lmin, vmax=lmax, cmap=plt.get_cmap('gray'))

plt.show()
