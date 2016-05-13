import numpy as np
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

s = 1000

a = 6*np.ones((s, s), dtype=np.float32)
b = np.ones((s, s), dtype=np.float32)
d = 2*np.ones((s, s), dtype=np.float32)
c = np.zeros((s, s), dtype=np.float32)

a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
d_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

f = open('add.cl', 'r')
defines = """
        #define SIZE %d
        """ % s
programName = defines + "".join(f.readlines())
program = cl.Program(ctx, programName).build()
program.add(queue, a.shape, None, a_buf, b_buf, d_buf, c_buf)

cl.enqueue_copy(queue, c, c_buf)
print(c)

from src import get_stamps
import dataproc as dp

basepath = "/media/Fran/data/"

folders = [#{"path": "sara/20130422/", "targets": [[677, 653], [462, 625]], "ap": 9, "sky": [12, 16],
           # "stamp": 20, "bias": "BiasNone001.fits", "dark": "DarkffNone001.fits", "flat": "Flat10Bessell R001.fits"},
           #{"path": "sara/20131108/", "targets": [[1167, 920], [1291, 1210]], "ap": 10, "sky": [12, 15],
           # "stamp": 20, "bias": "BiasNone001.fits", "dark": "DarksNone001.fits", "flat": "Flat10Bessell R001.fits"},
           {"path": "sara/20131117/", "targets": [[741, 684], [520, 286]], "ap": 5, "sky": [7, 9],
            "stamp": 15, "bias": "calib-20131118-None--bias-654.fits", "dark": "calib-20131118-None-604.fits",
            "flat": "flats-20131117-Bessell R-001.fits"},
           {"path": "sara/20131214/", "targets": [[520, 365], [434, 625]], "ap": 20, "sky": [25, 30],
            "stamp": 50, "bias": "Bias-011.fits", "dark": "Dark30s-000.fits", "flat": "Flat5-001.fits"},
           {"path": "sara/20140105/", "targets": [[498, 465], [417, 722]], "ap": 20, "sky": [25, 30],
            "stamp": 50, "bias": "bias-000.fits", "dark": "dark200s-000.fits", "flat": "flats1s000.fits"},
           {"path": "sara/20140118/", "targets": [[570, 269], [436, 539]], "ap": 12, "sky": [16, 20],
            "stamp": 30, "bias": "bias-000.fits", "dark": "dark200s-000.fits", "flat": "flats1s004.fits"}
           ]

for f in folders:
    f_path = basepath + f['path']
    print(f_path)
    io = dp.AstroDir(f_path + 'sci10/')
    dark = dp.AstroFile(f_path + 'dark/' + f['dark']).reader()
    bias = dp.AstroFile(f_path + 'bias/' + f['bias']).reader()
    flat = dp.AstroFile(f_path + 'flat/' + f['flat']).reader()

    sci_stamps, new_coords, stamp_coords, epoch, labels = get_stamps.get_stamps(io, f['targets'], f['stamp'])

    stamp_rad = f['stamp']

    for n in range(2):
        c = stamp_coords[n]
        c_full = new_coords[n]
        cx, cy = c[0][0], c[0][1]
        cxf, cyf = int(c_full[n][0]), int(c_full[n][1])
        bias_stamp = bias[(cxf-f['stamp']):(cxf+stamp_rad+1), (cyf-stamp_rad):(cyf+stamp_rad+1)]
        dark_stamp = dark[(cxf-f['stamp']):(cxf+stamp_rad+1), (cyf-stamp_rad):(cyf+stamp_rad+1)]
        flat_stamp = flat[(cxf-stamp_rad):(cxf+stamp_rad+1), (cyf-stamp_rad):(cyf+stamp_rad+1)]

