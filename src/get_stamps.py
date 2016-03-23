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


def get_stamps(sci, master_bias, master_dark, master_flat, target_coords, stamp_rad):
    """

    :param data:
    :param master_bias:
    :param master_dark:
    :param master_flat:
    :param target_coords: [[t1x, t1y], [t2x, t2y], ...]
    :param stamp_size:
    :return:
    """
    all_cubes = []
    data = sci.readdata()

    for c in target_coords:
        cx, cy = c[0], c[1]
        cube = []
        stamp = data[0][(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)]
        #print stamp
        cx0, cy0 = centroid(stamp)
        cx = cx - stamp_rad + cx0.round()
        cy = cy - stamp_rad + cy0.round()
        stamp = data[0][(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)]
        print(cx, cy)
        cube.append(stamp)
        for d in data[1:]:
            stamp = d[(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)]
            cx0, cy0 = centroid(stamp)
            cx = cx - stamp_rad + cx0.round()
            cy = cy - stamp_rad + cy0.round()
            cube.append(stamp)
        all_cubes.append(cube)

    return all_cubes


io = dp.AstroDir("/media/Fran/2011_rem/rawsci70/raw6")
# OJO coordenadas van Y,X
res = get_stamps(io, None, None, None, [[485, 737], [127, 521]], 30)
print(len(res), len(res[0]), res[0][0].shape)

import matplotlib.pyplot as plt

fig1 = plt.figure()
l = len(res[0])
n = len(res)

for i in range(1, l + 1):
    plt.subplot(2, l, i)
    lmin, lmax = zscale(res[0][i - 1])
    plt.imshow(res[0][i - 1], vmin=lmin, vmax=lmax, cmap=plt.get_cmap('gray'))
    plt.subplot(2, l, l + i)
    lmin, lmax = zscale(res[1][i - 1])
    plt.imshow(res[1][i - 1], vmin=lmin, vmax=lmax, cmap=plt.get_cmap('gray'))

plt.show()
