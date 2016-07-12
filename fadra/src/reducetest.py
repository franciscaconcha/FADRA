__author__ = 'fran'

import pyopencl as cl
import pyfits as pf
import numpy as np
import os
import time
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

size = 100

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

# TODO detectar memoria del device para reducir por bloques de imagenes
# devices[0].get_info(cl.global_mem_size)

from reductionTest import reductionTest

darks, flat, images, images_noisy = reductionTest.generate_images("../../reductionTest/config")

raw_names = []
t = 0

import shutil
folder1 = './reduce_test/'
folder2 = './sci_reduced/'
folder3 = './calib/'
for the_file in os.listdir(folder1):
    file_path = os.path.join(folder1, the_file)
    if os.path.isfile(file_path):
        os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
for the_file in os.listdir(folder2):
    file_path = os.path.join(folder2, the_file)
    if os.path.isfile(file_path):
        os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
for the_file in os.listdir(folder3):
    file_path = os.path.join(folder3, the_file)
    if os.path.isfile(file_path):
        os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)

for i in images_noisy:
    hdu = pf.PrimaryHDU(i)
    filename = "./reduce_test/reduced_" + "%03i.fits" % t
    hdu.writeto(filename)
    raw_names.append(filename)
    pf.open(filename)[0]
    t += 1
print("saved")

hdu = pf.PrimaryHDU(darks[0])
filename = "./calib/bias.fits"
hdu.writeto(filename)

import matplotlib.pyplot as plt

from reduction import GPUreduce, CPUreduce
import dataproc as dp
res2 = GPUreduce(dp.AstroDir("./reduce_test/"), "./sci_reduced/", dp.AstroDir("./calib/"), [darks[0]], [flat])
j = CPUreduce(dp.AstroDir("./reduce_test/"), "./sci_reduced/", dp.AstroDir("./calib/"), [darks[0]], [flat])

#for t in j:
#    t = np.round(t, 6)
"""
j = []
for t in j_dir:
    data = t.reader()
    j.append(data)

res2 = []
for t in res2_dir:
    data = t.reader()
    res2.append(data)
#print(j)"""

from CPUmath import mean_filter, median_filter

fig = plt.figure()
ax1 = plt.subplot(331)
ax1.set_title("Original")
plt.imshow(mean_filter(images_noisy[0], 3))
plt.subplot(334)
plt.imshow(images_noisy[1])
plt.subplot(337)
plt.imshow(images_noisy[2])

ax2 = plt.subplot(332)
ax2.set_title("GPU")
l, u = zscale(res2[0])
plt.imshow(res2[0], vmin=l, vmax=u)
plt.subplot(335)
l, u = zscale(res2[1])
plt.imshow(res2[1], vmin=l, vmax=u)
plt.subplot(338)
l, u = zscale(res2[2])
plt.imshow(res2[2], vmin=l, vmax=u)

ax3 = plt.subplot(333)
ax3.set_title("CPU")
l, u = zscale(j[0])
plt.imshow(j[0], vmin=l, vmax=u)
plt.subplot(336)
l, u = zscale(j[1])
plt.imshow(j[1], vmin=l, vmax=u)
plt.subplot(339)
l, u = zscale(j[2])
plt.imshow(j[2], vmin=l, vmax=u)

plt.show()