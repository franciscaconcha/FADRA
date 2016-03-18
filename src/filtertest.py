__author__ = 'fran'

import pyfits as pf
import numpy as np

size = 3

def zscale(img,  trim = 0.05, contr=1, mask=None):
    """Returns lower and upper limits found by zscale algorithm for improved contrast in astronomical images.

:param mask: bool ndarray
    True are good pixels, pixels marked as False are ignored
:rtype: (min, max)
    Minimum and maximum values recommended by zscale
"""
    import scipy as sp
    if not isinstance(img, sp.ndarray):
        img = sp.array(img)
    if mask is None:
        mask = (sp.isnan(img) == False)

    itrim = int(img.size*trim)
    x = sp.arange(mask.sum()-2*itrim)+itrim

    sy = sp.sort(img[mask].flatten())[itrim:img[mask].size-itrim]
    a, b = sp.polyfit(x, sy, 1)

    return b, a*img.size/contr+b

import matplotlib.pyplot as plt

from reduction import CPUreduce
from CPUmath import mean_filter, median_filter
import dataproc as dp

#j = CPUreduce(dp.AstroDir("./reduce_test/"), "./sci_reduced/", dp.AstroDir("./calib/"), [darks[0]], [flat])

im = pf.open("./reduce_test/reduced_002.fits")
im_data = im[0].data

fig = plt.figure()
ax1 = plt.subplot(231)
ax1.set_title("Original")
l, u = zscale(im_data)
plt.imshow(im_data, vmin=l, vmax=u)

ax2 = plt.subplot(232)
title = "Mine\nMean, " + "%d px window" % size
ax2.set_title(title)
im_mean = mean_filter(im_data, size)
l, u = zscale(im_mean)
plt.imshow(im_mean, vmin=l, vmax=u)

ax3 = plt.subplot(233)
title = "Mine\nMedian, " + "%d px window" % size
ax3.set_title(title)
im_median = median_filter(im_data, size)
l, u = zscale(im_median)
plt.imshow(im_median, vmin=l, vmax=u)

import scipy.signal, scipy.ndimage

ax1 = plt.subplot(234)
ax1.set_title("Original")
l, u = zscale(im_data)
plt.imshow(im_data, vmin=l, vmax=u)

ax2 = plt.subplot(235)
title = "Scipy\nMean, " + "%d px window" % size
ax2.set_title(title)
kernel = np.ones((im_data.shape))/(size*size)
#im_mean = scipy.ndimage.filters.convolve(im_data, kernel)
im_mean = scipy.signal.convolve(im_data, kernel)
l, u = zscale(im_mean)
plt.imshow(im_mean, vmin=l, vmax=u)

ax3 = plt.subplot(236)
title = "Scipy\nMedian, " + "%d px window" % size
ax3.set_title(title)
im_median = scipy.signal.medfilt(im_data, size)
l, u = zscale(im_median)
plt.imshow(im_median, vmin=l, vmax=u)

plt.show()