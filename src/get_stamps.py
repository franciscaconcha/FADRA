import dataproc as dp
import copy
import scipy as sp
import time
import numpy as np

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
    """Polynomial fit for sky subtraction

    :param coef: sky fit polynomial coefficients
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

    iy, ix = sp.mgrid[0:len(arr), 0:len(arr)]

    cy = sp.sum(iy * arr) / sp.sum(arr)
    cx = sp.sum(ix * arr) / sp.sum(arr)

    return cy, cx


def get_stamps(sci, target_coords, stamp_rad):
    """

    :param sci:
    :type sci: AstroDir
    :param target_coords: [[t1x, t1y], [t2x, t2y], ...]
    :param stamp_rad:
    :return:
    """

    data = sci.files

    all_cubes = []
    #data = sci.readdata()
    epoch = sci.getheaderval('DATE-OBS')
    #epoch = sci.getheaderval('MJD-OBS')
    labels = sci.getheaderval('OBJECT')
    new_coords = []
    stamp_coords =[]

    import pyfits as pf

    for tc in target_coords:
        cube, new_c, st_c = [], [], []
        cx, cy = tc[0], tc[1]
        for df in data:
            dlist = pf.open(df.filename)
            d = dlist[0].data
            #d = df.reader({'rawdata': True})
            stamp = d[cx - stamp_rad:cx + stamp_rad + 1, cy - stamp_rad:cy + stamp_rad +1]
            cx_s, cy_s = centroid(stamp)
            cx = cx - stamp_rad + cx_s.round()
            cy = cy - stamp_rad + cy_s.round()
            stamp = d[cx - stamp_rad:cx + stamp_rad + 1, cy - stamp_rad:cy + stamp_rad +1]
            cube.append(stamp)
            st_c.append([cx_s.round(), cy_s.round()])
            new_c.append([cx, cy])
            dlist.close()
        all_cubes.append(cube)
        new_coords.append(new_c)
        stamp_coords.append(st_c)

    return all_cubes, new_coords, stamp_coords, epoch, labels[-2:]


def CPUphot(sci, dark, flat, coords, stamp_coords, ap, sky, stamp_rad, deg=1, gain=None, ron=None):
    n_targets = len(sci)
    n_frames = len(sci[0])
    all_phot = []
    all_err = []

    t0 = time.clock()

    for n in range(n_targets):  # For each target
        target = sci[n]
        c = stamp_coords[n]
        c_full = coords[n]
        t_phot, t_err = [], []
        for t in range(n_frames):
            cx, cy = c[0][0], c[0][1]  # TODO ojo con esto
            cxf, cyf = int(c_full[t][0]), int(c_full[t][1])
            cs = [cy, cx]

            # Reduction!
            # Callibration stamps are obtained using coordinates from the "full" image
            dark_stamp = dark[(cxf-stamp_rad):(cxf+stamp_rad+1), (cyf-stamp_rad):(cyf+stamp_rad+1)]
            flat_stamp = flat[(cxf-stamp_rad):(cxf+stamp_rad+1), (cyf-stamp_rad):(cyf+stamp_rad+1)]
            data = (target[t] - dark_stamp) / flat_stamp

            # Photometry!
            d = centraldistances(data, cs)
            dy, dx = data.shape
            y, x = sp.mgrid[-cs[0]:dy - cs[0], -cs[1]:dx - cs[1]]

            # Compute sky correction
            # Case 1: sky = [fit, map_of_sky_pixels]
            if isinstance(sky[0], sp.ndarray):
                fit = sky[0]
                idx = sky[1]

            # Case 2: sky = [inner_radius, outer_radius]
            else:
                import scipy.optimize as op
                idx = (d > sky[0]) * (d < sky[1])
                errfunc = lambda coef, x, y, z: (bipol(coef, x, y) - z).flatten()
                coef0 = sp.zeros((deg, deg))
                coef0[0, 0] = data[idx].mean()
                fit, cov, info, mesg, success = op.leastsq(errfunc, coef0.flatten(), args=(x[idx], y[idx], data[idx]), full_output=1)

            # Apply sky correction
            n_pix_sky = idx.sum()
            sky_fit = bipol(fit, x, y)
            sky_std = (data-sky_fit)[idx].std()
            res = data - sky_fit  # minus sky

            res2 = res[d < ap*4].ravel()
            d2 = d[d < ap*4].ravel()

            tofit = lambda d, h, sig: h*dp.gauss(d, sig, ndim=1)

            import scipy.optimize as op
            try:
                sig, cov = op.curve_fit(tofit, d2, res2, sigma=1/sp.sqrt(sp.absolute(res2)), p0=[max(res2), ap/3])
            except RuntimeError:
                sig = sp.array([0, 0, 0])

            fwhmg = 2.355*sig[1]

            #now photometry
            phot = float(res[d < ap].sum())
            #print("phot: %.5d" % (phot))

            #now the error
            if gain is None:
                error = None
            else:
                n_pix_ap = res[d < ap].sum()
                error = phot_error(phot, sky_std, n_pix_ap, n_pix_sky, gain, ron=ron)

            t_phot.append(phot)
            t_err.append(error)
        all_phot.append(t_phot)
        all_err.append(t_err)

    t1 = time.clock() - t0

    import TimeSeries as ts
    return ts.TimeSeries(all_phot, all_err, None), t1


def GPUphot(sci, dark, flat, coords, stamp_coords, ap, sky, stamp_rad, deg=1, gain=None, ron=None):
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

    n_targets = len(sci)
    all_phot = []
    all_err = []

    t0 = time.clock()

    for n in range(n_targets):  # For each target
        target = np.array(sci[n])
        c = stamp_coords[n]
        c_full = coords[n]
        cx, cy = c[0][0], c[0][1]
        cxf, cyf = int(c_full[n][0]), int(c_full[n][1])
        dark_stamp = dark[(cxf-stamp_rad):(cxf+stamp_rad+1), (cyf-stamp_rad):(cyf+stamp_rad+1)]
        flat_stamp = flat[(cxf-stamp_rad):(cxf+stamp_rad+1), (cyf-stamp_rad):(cyf+stamp_rad+1)]

        flattened_dark = dark_stamp.flatten()
        dark_f = flattened_dark.reshape(len(flattened_dark))

        flattened_flat = flat_stamp.flatten()
        flat_f = flattened_flat.reshape(len(flattened_flat))

        this_phot, this_error = [], []

        for f in target:
            """phoot = np.zeros((4,))
            for i in range(f.shape[0]):
                for j in range(f.shape[1]):
                    dist = np.sqrt((i-cx)**2 + (j-cy)**2)
                    if dist < ap:
                        phoot[0] += (f[i, j]-dark_stamp[i, j])/flat_stamp[i, j]
                        phoot[1] += 1
                    elif dist > sky[0] and dist < sky[1]:
                        phoot[2] += (f[i, j]-dark_stamp[i, j])/flat_stamp[i, j]
                        phoot[3] += 1
            res_val = (phoot[0] - (phoot[2]/phoot[3])*phoot[1])
            this_phot.append(phoot[0] - (phoot[2]/phoot[3])*phoot[1])"""

            s = f.shape
            ss = s[0] * s[1]
            ft = f.reshape(1, ss)

            target_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ft[0])
            dark_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dark_f)
            flat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_f)
            res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, np.zeros((4, ), dtype=np.int32).nbytes)

            f_cl = open('/media/Fran/fractal/src/photometry.cl', 'r') #TODO ojo con este path
            defines = """
                    #define n %d
                    #define centerX %d
                    #define centerY %d
                    #define aperture %d
                    #define sky_inner %d
                    #define sky_outer %d
                    #define SIZE %d
                    """ % (2*stamp_rad+1, cx, cy, ap, sky[0], sky[1], f.shape[0])
            programName = defines + "".join(f_cl.readlines())

            program = cl.Program(ctx, programName).build()
            #queue, global work group size, local work group size
            program.photometry(queue, ft[0].shape, #(512, 512, 1), #np.zeros((2*stamp_rad, 2*stamp_rad, 1)).shape,
                               None, #(1, 1, 1), #np.zeros((512)).shape,
                               target_buf, dark_buf, flat_buf, res_buf)

            res = np.zeros((4, ), dtype=np.int32)
            cl.enqueue_copy(queue, res, res_buf)

            res_val = (res[0] - (res[2]/res[3])*res[1])/10000
            this_phot.append(res_val)

            #now the error
            if gain is None:
                error = None
            else:
                d = centraldistances(f, [cx, cy])
                sky_std = f[(d > sky[0]) & (d < sky[1])].std()
                error = phot_error(res_val, sky_std, res[1], res[3], gain, ron=ron)/1000
                #error = phot_error(res_val, sky_std, phoot[1], phoot[3], gain, ron=ron)
            this_error.append(error)

        all_phot.append(this_phot)
        all_err.append(this_error)

    t1 = time.clock() - t0

    import TimeSeries as ts
    return ts.TimeSeries(all_phot, all_err, None), t1


def photometry(sci, mbias, mdark, mflat, target_coords, aperture, stamp_rad, sky, deg=1, gain=None, ron=None, gpu=False):
    sci_stamps, new_coords, stamp_coords, epoch, labels = get_stamps(sci, target_coords, stamp_rad)

    print("Stamps done")

    if gpu:
        ts, tt = GPUphot(sci_stamps, mdark-mbias, mflat-mbias, new_coords, stamp_coords, aperture, sky, stamp_rad, deg, gain, ron)
    else:
        ts, tt = CPUphot(sci_stamps, mdark-mbias, mflat-mbias, new_coords, stamp_coords, aperture, sky, stamp_rad, deg, gain, ron)

    ts.set_epoch(epoch)
    labels[1] = 'REF1'
    ts.set_labels(labels)
    return ts, tt


"""io = dp.AstroDir("/media/Fran/data/sara/20131214/sci10")
#io = dp.AstroDir("/media/Fran/data/dk154/d1/sci/WASP26")
# OJO coordenadas van Y,X
#res = get_stamps(io, None, None, None, [[577, 185], [488, 739]], 20)
import numpy as np
#dark = np.zeros((2048,2048))
#bias = np.zeros((2048,2048))
#flat = np.ones((2048,2048))

dark = dp.AstroFile("/media/Fran/data/sara/20131214/dark/Dark30s-000.fits")
bias = dp.AstroFile("/media/Fran/data/sara/20131214/bias/Bias-011.fits")
flat = dp.AstroFile("/media/Fran/data/sara/20131214/flat/Flat5-001.fits")

# estas coordenadas ya vienen en formato y, x
target_coords = [[520, 365], [434, 625], [536, 563]]
#target_coords =[[1077, 1022], [412, 1505]]
aperture = 20
stamp_rad = 50
sky_i = 25
sky_o = 30
sky = [sky_i, sky_o]
gain = 2.0
ron = 14.0

import time
gpu_time_0 = time.clock()
res_gpu, gpu_time = photometry(io, bias, dark, flat, target_coords, aperture, stamp_rad, sky, gain=gain, ron=ron, gpu=True)
#gpu_time = time.clock() - gpu_time_0

cpu_time_0 = time.clock()
res_cpu, cpu_time = photometry(io, bias, dark, flat, target_coords, aperture, stamp_rad, sky, gain=gain, ron=ron)
#cpu_time = time.clock() - cpu_time_0

print("CPU: %.2f s || GPU: %.2f s" % (cpu_time, gpu_time))

res_cpu.plot()
res_gpu.plot()"""


"""from sklearn.metrics import mean_squared_error
from math import sqrt

rc = [int(i) for i in res_cpu[0]]

rms = np.array(sqrt(mean_squared_error(rc, res_gpu[0])))
print rms
print rms/(max(np.array(res_cpu[0])) - min(np.array(res_gpu[0])))

#print np.array(rc) - np.array(res_gpu[0])

#print max(np.array(rc) - np.array(res_gpu[0]))/max(np.array(res_cpu[0]))"""

"""import matplotlib.pyplot as plt
import dataproc as dp
fig, ax, cpu_epoch = dp.axesfig_xdate(None, res_cpu.epoch)
ax.errorbar(cpu_epoch, res_cpu[0], marker='o', label='CPU')
ax.errorbar(cpu_epoch, res_gpu[0], marker='o', label='GPU')
plt.show()"""

"""from dataproc.timeseries import astrointerface
print io.readdata().shape
interface = astrointerface.AstroInterface(io.readdata()[0])
interface.execute()"""

"""sci_stamps, new_coords, stamp_coords, epoch, labels = get_stamps(io, target_coords, stamp_rad)
l = len(sci_stamps[0])
n = 2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(25, 5))

grid = gridspec.GridSpec(2, l, wspace=0.0, hspace=0.5)
for i in range(1, l + 1):
    ax1 = plt.Subplot(fig, grid[i - 1])
    ax1.imshow(sci_stamps[0][i - 1], cmap=plt.get_cmap('gray'))
    circle1=plt.Circle((stamp_coords[0][i - 1][0], stamp_coords[0][i - 1][1]), aperture, color='r',fill=False)
    circlesi1=plt.Circle((stamp_coords[0][i - 1][0], stamp_coords[0][i - 1][1]), sky_i, color='g',fill=False)
    circleso1=plt.Circle((stamp_coords[0][i - 1][0], stamp_coords[0][i - 1][1]), sky_o, color='g',fill=False)
    ax1.add_artist(circle1)
    ax1.add_artist(circlesi1)
    ax1.add_artist(circleso1)
    #ax1.plot(20, 20, 'or')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2 = plt.Subplot(fig, grid[l + i - 1])
    ax2.imshow(sci_stamps[1][i - 1], cmap=plt.get_cmap('gray'))
    circle2=plt.Circle((stamp_coords[1][i - 1][0],stamp_coords[1][i - 1][1]),aperture,color='r',fill=False)
    circlesi2=plt.Circle((stamp_coords[1][i - 1][0], stamp_coords[1][i - 1][1]), sky_i, color='g',fill=False)
    circleso2=plt.Circle((stamp_coords[1][i - 1][0], stamp_coords[1][i - 1][1]), sky_o, color='g',fill=False)
    ax2.add_artist(circle2)
    ax2.add_artist(circlesi2)
    ax2.add_artist(circleso2)
    #ax2.plot(20, 20, 'or')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)
#plt.show()
plt.savefig('../../foo.png', bbox_inches='tight')"""