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
    all_cubes = []
    data = sci.readdata()
    epoch = sci.getheaderval('MJD-OBS')
    labels = sci.getheaderval('OBJECT')
    new_coords = []
    stamp_coords =[]

    for c in target_coords:
        cx, cy = c[0], c[1]
        cube, new_c, st_c = [], [], []
        stamp = np.array(data[0][(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)])
        cx0, cy0 = centroid(stamp)
        st_c.append([cx0.round(), cy0.round()])
        cx = cx - stamp_rad + cx0.round()
        cy = cy - stamp_rad + cy0.round()
        stamp = np.array(data[0][(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)])
        cube.append(stamp)
        new_c.append([cx, cy])
        for d in data[1:]:
            stamp = np.array(d[(cx-stamp_rad):(cx+stamp_rad+1), (cy-stamp_rad):(cy+stamp_rad+1)])
            cx0, cy0 = centroid(stamp)
            st_c.append([cx0.round(), cy0.round()])
            cx = cx - stamp_rad + cx0.round()
            cy = cy - stamp_rad + cy0.round()
            cube.append(stamp)
            new_c.append([cx, cy])
        all_cubes.append(cube)
        new_coords.append(new_c)
        stamp_coords.append(st_c)

    return all_cubes, new_coords, stamp_coords, epoch, labels[-2:]


def CPUphot(sci, dark, flat, coords, stamp_coords, ap, sky, stamp_rad, deg=1, gain=None, ron=None):
    n_targets = len(sci)
    n_frames = len(sci[0])
    all_phot = []
    all_err = []
    for n in range(n_targets):  # For each target
        target = sci[n]
        c = stamp_coords[n]
        c_full = coords[n]
        t_phot, t_err = [], []
        for t in range(n_frames):
            cx, cy = c[0][0], c[0][1]  # TODO ojo con esto
            cxf, cyf = c_full[t][0], c_full[t][1]
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
            print("phot: %.5d" % (phot))

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

    import TimeSeries as ts
    return ts.TimeSeries(all_phot, all_err, None)


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
    n_frames = len(sci[0])
    all_phot = []
    all_err = []
    for n in range(n_targets):  # For each target
        target = np.array(sci[n])
        c = stamp_coords[n]
        c_full = coords[n]
        t_phot, t_err = [], []
        cx, cy = c[0][0], c[0][1]
        cxf, cyf = c_full[n][0], c_full[n][1]
        dark_stamp = dark[(cxf-stamp_rad):(cxf+stamp_rad+1), (cyf-stamp_rad):(cyf+stamp_rad+1)]
        flat_stamp = flat[(cxf-stamp_rad):(cxf+stamp_rad+1), (cyf-stamp_rad):(cyf+stamp_rad+1)]

        target_flat = []
        for f in target:
            s = f.shape
            ss = s[0] * s[1]
            ft = f.reshape(1, ss)
            target_flat.append(ft[0])

        target_f = np.array([item for sublist in target_flat for item in sublist])
        #print len(target_flat), len(target_f)
        #target_f = target_flat.reshape(1, len(target_flat))

        flattened_dark = dark_stamp.flatten()
        dark_f = flattened_dark.reshape(1, len(flattened_dark))

        flattened_flat = flat_stamp.flatten()
        flat_f = flattened_flat.reshape(1, len(flattened_flat))

        target_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_f)
        dark_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dark_f[0])
        flat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_f[0])
        res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, np.zeros((1, len(target_flat)), dtype=np.float32).nbytes)

        f = open('photometry.cl', 'r')
        defines = """
                    #define n %d
                    #define centerX %d
                    #define centerY %d
                    #define aperture %d
                    #define sky_inner %d
                    #define sky_outer %d
                """ % (2*stamp_rad, cx, cy, ap, sky[0], sky[1])
        programName = defines + "".join(f.readlines())

        program = cl.Program(ctx, programName).build()
        program.photometry(queue, target_f.shape, None, target_buf, dark_buf, flat_buf, res_buf)  # sizeX, sizeY, sizeZ

        res = np.zeros((1, len(target_flat)), dtype=np.float32)
        cl.enqueue_copy(queue, res, res_buf)
        all_phot.append(res)
        all_err.append(res)

    print all_phot
    import TimeSeries as ts
    return ts.TimeSeries(all_phot, all_err, None)


def photometry(sci, mbias, mdark, mflat, target_coords, aperture, stamp_rad, sky, gpu=False):
    sci_stamps, new_coords, stamp_coords, epoch, labels = get_stamps(sci, target_coords, stamp_rad)
    print epoch

    if gpu:
        ts = GPUphot(sci_stamps, mdark-mbias, mflat-mbias, new_coords, stamp_coords, aperture, sky, stamp_rad)
    else:
        ts = CPUphot(sci_stamps, mdark-mbias, mflat-mbias, new_coords, stamp_coords, aperture, sky, stamp_rad)

    ts.set_epoch(epoch)
    labels[1] = 'REF1'
    ts.set_ids(labels)
    return ts


io = dp.AstroDir("/media/Fran/2011_rem/rawsci70/raw6")
# OJO coordenadas van Y,X
#res = get_stamps(io, None, None, None, [[577, 185], [488, 739]], 20)
import numpy as np
dark = np.zeros(io.readdata()[0].shape)
bias = np.zeros(io.readdata()[0].shape)
flat = np.ones(io.readdata()[0].shape)

res = photometry(io, bias, dark, flat, [[577, 185], [488, 739]], 8, 50, [10, 15])#, gpu=True)
#print(len(res[0]), res[0][0].shape)
#print res[1]

res.plot()


"""for i in range(1, l + 1):
    plt.subplot(n, l, i)
    #lmin, lmax = zscale(res.channels[0][i - 1])
    plt.imshow(res.channels[0][i - 1])#, vmin=lmin, vmax=lmax, cmap=plt.get_cmap('gray'))
    plt.subplot(n, l, l + i)
    #lmin, lmax = zscale(res.channels[1][i - 1])
    plt.imshow(res.channels[1][i - 1])#, vmin=lmin, vmax=lmax, cmap=plt.get_cmap('gray'))

plt.show()"""
