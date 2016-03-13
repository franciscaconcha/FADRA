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
        mask = (sp.isnan(img)==False)

    itrim = int(img.size*trim)
    x = sp.arange(mask.sum()-2*itrim)+itrim

    sy = sp.sort(img[mask].flatten())[itrim:img[mask].size-itrim]
    a, b = sp.polyfit(x, sy, 1)

    return b, a*img.size/contr+b

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

# TODO detectar memoria del device para reducir por bloques de imagenes
# devices[0].get_info(cl.global_mem_size)

from reductionTest import reductionTest

darks, flat, images, images_noisy = reductionTest.generate_images("../../reductionTest/config")

raw_names = []
t = 0
for i in images_noisy:
    #hdu = pf.PrimaryHDU(i)
    filename = "./reduce_test/reduced_" + "%03i.fits" % t
    #hdu.writeto(filename)
    raw_names.append(filename)
    pf.open(filename)[0]
    t += 1
print("saved")

import matplotlib.pyplot as plt

img = np.array([])
cpu_img = []
i = 0
ss = 0

for i in images_noisy:
    sh = i.shape
    ss = sh[0] * sh[1]
    data = i.reshape(1, ss)
    ndata = data[0]
    img = np.append(img, ndata)

print(len(images_noisy))
print(images_noisy[0].shape)
print("**")

#print(img.shape)

dark = darks[0].reshape(1, ss)
dark2 = np.append(np.append(dark[0], dark[0]), dark[0])
print(len(dark2))

flat = flat.reshape(1, ss)
print(len(flat[0]))

img_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
dark_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dark[0])
flat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat[0])
res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)

f = open('reduce.cl', 'r')
programName = "".join(f.readlines())

program = cl.Program(ctx, programName).build()

start = time.clock()
program.reduce(queue, img.shape, None, dark_buf, flat_buf, img_buf, res_buf) #sizeX, sizeY, sizeZ
end = time.clock()
print("GPU: %f s" % (end - start))

res = np.empty_like(img)
cl.enqueue_copy(queue, res, res_buf)
res2 = np.reshape(res, (3, size, size))
print(res2.shape)
print(len(res2))

#j = []
#for i in images_noisy:
#    j.append((i - darks[0])/flat[0].reshape(size, size))

from reduction import CPUreduce
import dataproc as dp
j_dir = CPUreduce(dp.AstroDir("./reduce_test/"), "./sci_reduced/", darks[0], darks[0], flat[0])

#for t in j:
#    t = np.round(t, 6)

j = []
for t in j_dir:
    data = t.reader()
    j.append(data)

fig = plt.figure()
ax1 = plt.subplot(331)
ax1.set_title("Original")
plt.imshow(images_noisy[0])
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

"""
print("Done reading image files.")

dcurr = pf.open(path + "dark/D025N000.fits")
cpu_dark = dcurr[0].data
dcurr.close()
dsh = cpu_dark.shape
darkdataarray = cpu_dark.reshape(1, dsh[0]*dsh[1])
dark = np.array(darkdataarray)

fcurr = pf.open(path + "flat/F025I001.fits")
cpu_flat = fcurr[0].data
fcurr.close()
fsh = cpu_flat.shape
flatdataarray = cpu_flat.reshape(1, fsh[0]*fsh[1])
flat = np.array(flatdataarray)

print("Reducing %d images." % (len(cpu_img)))

f3=[]

#CPU reduction
start = time.clock()
for f in cpu_img:
    f2 = (f - cpu_dark)/cpu_flat
    f3.append(f2)
end = time.clock()
print("CPU: %f s" % (end - start))

#GPU reduction
img_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
dark_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dark)
flat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)

f = open('reduce.cl', 'r')
programName = "".join(f.readlines())

program = cl.Program(ctx, programName).build()

start = time.clock()
program.reduce(queue, img.shape, None, dark_buf, flat_buf, img_buf, res_buf) #sizeX, sizeY, sizeZ
end = time.clock()
print("GPU: %f s" % (end - start))

res = np.empty_like(img)
cl.enqueue_copy(queue, res, res_buf)"""