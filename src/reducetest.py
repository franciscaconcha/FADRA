__author__ = 'fran'

import pyopencl as cl
import pyfits as pf
import numpy as np
import os
import time
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

f = open("dataset", "r")
ffiles = f.readlines()
path = "/media/Fran/2011_rem/"

img = np.array([])
cpu_img = []
i = 0

#n_imgs = 100

for f in ffiles:
    curr = pf.open(path + "allsci/" + f[:-1])
    scidata = curr[0].data
    curr.close()
    sh = scidata.shape
    #print(sh)
    if sh != (1024, 1024):
        continue
    else:
        cpu_img.append(scidata)
        ss = sh[0] * sh[1]
        data = scidata.reshape(1, ss)
        ndata = data[0]
        img = np.append(img, ndata)
        i += 1
        if i % 50 == 0:
            print("Read %d files" % i)
        if i == 110:
            break

print("Done reading image files.")

dcurr = pf.open(path + "dark/20110127/D025N000.fits")
cpu_dark = dcurr[0].data
dcurr.close()
dsh = cpu_dark.shape
darkdataarray = cpu_dark.reshape(1, dsh[0]*dsh[1])
dark = np.array(darkdataarray)

fcurr = pf.open(path + "flat/20110127/F025I001.fits")
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
cl.enqueue_copy(queue, res, res_buf)