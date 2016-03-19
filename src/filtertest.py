__author__ = 'fran'

import pyfits as pf
import numpy as np

size = 5

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

# My CPU algorithms
fig = plt.figure()
ax1 = plt.subplot(331)
ax1.set_title("Original")
l, u = zscale(im_data)
plt.imshow(im_data)#, vmin=l, vmax=u)

ax2 = plt.subplot(332)
title = "Mine\nMean, " + "%d px window" % size
ax2.set_title(title)
im_mean = mean_filter(im_data, size)
l, u = zscale(im_mean)
plt.imshow(im_mean)#, vmin=l, vmax=u)

ax3 = plt.subplot(333)
title = "Mine\nMedian, " + "%d px window" % size
ax3.set_title(title)
im_median = median_filter(im_data, size)
l, u = zscale(im_median)
plt.imshow(im_median)#, vmin=l, vmax=u)

import scipy.signal, scipy.ndimage

# Scipy filters
ax1 = plt.subplot(334)
ax1.set_title("Original")
l, u = zscale(im_data)
plt.imshow(im_data)#, vmin=l, vmax=u)

ax2 = plt.subplot(335)
title = "Scipy\nMean, " + "%d px window" % size
ax2.set_title(title)
kernel = np.ones((im_data.shape))/(size*size)
im_mean = scipy.ndimage.filters.uniform_filter(im_data, size, mode='constant')
#im_mean = scipy.signal.convolve(im_data, kernel)
l, u = zscale(im_mean)
plt.imshow(im_mean)#, vmin=l, vmax=u)
print(im_data.__class__)
print(im_mean[98])

ax3 = plt.subplot(336)
title = "Scipy\nMedian, " + "%d px window" % size
ax3.set_title(title)
im_median = scipy.signal.medfilt(im_data, size)
l, u = zscale(im_median)
plt.imshow(im_median)#, vmin=l, vmax=u)

# My GPU algorithms
ax4 = plt.subplot(337)
ax4.set_title("Original")
l, u = zscale(im_data)
plt.imshow(im_data)#, vmin=l, vmax=u)

import pyopencl as cl
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

platforms = cl.get_platforms()
if len(platforms) == 0:
    print("Failed to find any OpenCL platforms.")

devices = platforms[0].get_devices(cl.device_type.GPU)
if len(devices) == 0:
    print("Could not find GPU device, trying CPU...")
    devices = platforms[0].get_devices(cl.device_type.CPU)
    if len(devices) == 0:
        print("Could not find OpenCL GPU or CPU device.")

ctx = cl.Context([devices[0]])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

kernel = (1/(size*size))*np.ones((size, size))

img_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=im_data)
kernel_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel)
res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, im_data.nbytes)

im_size = im_data.shape[0]
hfs = np.floor(size/2)

f = open('convolve.cl', 'r')
defines = """
    #define IMAGE_W %d
    #define IMAGE_H %d
    #define FILTER_SIZE %d
    #define HALF_FILTER_SIZE %d
    #define TWICE_HALF_FILTER_SIZE %d
    #define HALF_FILTER_SIZE_IMAGE_W %d
    """ % (im_size, im_size, size, hfs, 2*hfs, im_size*hfs)
programName = defines + "".join(f.readlines())

program = cl.Program(ctx, programName).build()
program.convolve(queue, im_data.shape, None, img_buf, kernel_buf, res_buf)

res = np.empty_like(im_data)
cl.enqueue_copy(queue, res, res_buf)


ax4 = plt.subplot(338)
ax4.set_title("GPU")
l, u = zscale(res)
plt.imshow(res)#, vmin=l, vmax=u)

plt.show()