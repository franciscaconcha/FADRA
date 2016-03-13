__author__ = 'fran'

import CPUmath
import pyfits as pf
import warnings
from functools import wraps as _wraps
import numpy as np
from dataproc.core import io_file, io_dir


def _check_combine_input(function):
    """ Decorator that checks all images in the input are
        of the same size and type. Else, raises an error.
    """

    @_wraps(function)
    def input_checker(instance, *args, **kwargs):
        try:
            init_shape = instance[0].shape
        except AttributeError:
            raise IOError("First file to be combined in CPUmath.%s is not a supported type." % function.__name__)

        init_type = type(instance[0])

        for i in instance:
            try:
                i_shape = i.shape
            except AttributeError:
                raise IOError("File to be combined in CPUmath.%s is not a supported type." % function.__name__)

            if i_shape == init_shape and type(i) is init_type:
                continue
            else:
                if i_shape != init_shape:
                    instance.remove(i)
                    warnings.warn("File %s is of different shape. Will be ignored in %s." % (i, function.__name__))
                    # raise IOError("Files to be combined in CPUmath.%s are not all of the same shape." % function.__name__)
                else:
                    raise IOError(
                        "Files to be combined in CPUmath.%s are not all of the same type." % function.__name__)
        return function(instance, *args, **kwargs)

    return input_checker


@_check_combine_input
def get_masterbias(bias, combine_mode, save):
    """ Returns masterbias, combining all bias files using the given function.
        If not, uses CPU mean as default. Returns masterbias file and brings
        option to save it to .fits file.
    :param bias: np.ndarray with all bias arrays
    :param combine_mode: function used to combine bias
    :param save: true if want to save the master bias file. Default is false.
    :return: np.ndarray (master bias)
    """
    master_bias = combine_mode(bias)

    if save:
        hdu = pf.PrimaryHDU(master_bias)
        hdu.writeto('MasterBias.fits')

    print("Masterbias done")
    return master_bias


@_check_combine_input
def get_masterflat(flats, combine_mode, save):
    """ Returns masterflat, combining all flat files using the given function.
        If not, uses CPU mean as default. Returns masterflat file and brings
        option to save it to .fits file.
    :param flats: np.ndarray with all flat arrays
    :param combine_mode: function used to combine flats
    :param save: true if want to save the master flat file. Default is false.
    :return: np.ndarray (master flat)
    """
    master_flat = combine_mode(flats)

    if save:
        hdu = pf.PrimaryHDU(master_flat)
        hdu.writeto('MasterFlat.fits')

    print("MasterFlat done")
    return master_flat


def column(array, x, y):
    """ Returns vertical column from stack of 2D arrays.
    :param array: Stack (list) of 2D arrays
    :param x: x coordinate of desired column
    :param y: y coordinate of desired column
    :return: list
    """
    l = len(array)
    c = []
    for i in range(l):
        c.append(array[i][x][y])
    return c


# Interpolates darks to necessary exposure time
def interpol(darks, time):
    times = []
    for f in darks:
        d = pf.open(f)
        t = d[0].header['EXPTIME']
        d.close()
        if d == time:
            return f
        else:
            times.append(t)

    # If it didn't find a dark with desired exposure time, it interpolates one column by column
    d = pf.open(darks[0])
    s = d.shape
    md = np.zeros(s)
    d.close()

    for i in range(s[0]):
        for j in range(s[1]):
            md = sp.interp(time, times, column(darks, i, j))

    return md


@_check_combine_input
def get_masterdark(darks, combine_mode, time, save):
    """ Returns masterdark, combining all dark files using the given function.
        If not, uses CPU mean as default. Returns masterdark file and brings
        option to save it to .fits file.
    :param flats: np.ndarray with all dark arrays
    :param combine_mode: function used to combine darks
    :param save: true if want to save the master dark file. Default is false.
    :return: np.ndarray (master dark)
    """
    if time is None:
        master_dark = combine_mode(darks)
    else:
        master_dark = None
        for d in darks:
            exp_time = pf.getheader(d)['EXPTIME']
            if exp_time == time:
                master_dark = d

        if master_dark is None:
            master_dark = interpol(darks, time)

    if save:
        hdu = pf.PrimaryHDU(master_dark)
        hdu.writeto('MasterDark.fits')

    print("MasterDark done")
    return master_dark


def CPUreduce(raw, sci_path, bias=None, dark=None, flat=None,
              combine_mode=CPUmath.mean_combine, exp_time=None, save_masters=False):
    """ Reduces image files. Master calibration fields should be attached to the raw AstroDir.
        If AstroDirs are given for bias, dark, or flat, they are combined to obtain masters.
        Combination function given in combine_mode, default is CPU mean_combine.
        Saves masters to their AstroDir paths if save_masters = True.
        Returns AstroDir of reduced sci images.
        Saves reduced images to sci AstroDir path is save_reduced = True.
    :param raw: AstroDir of raw files, with corresponding masters attached
    :param sci_path: Path to save reduced files
    :param bias: (optional) AstroDir of all bias files to be combined
    :param flat: (optional) AstroDir of all flat files to be combined
    :param dark: (optional) AstroDir of all dark files to be combined
    :param combine_mode: (optional) function used to combine bias, darks, and flats
    :param exp_time: (optional) desired exposure time. If a value is given, the dark AstroDir will be
                    searched for the corresponding dark for said exposure time, and that dark will be
                    used as MasterDark. If no dark is found for that exposure time, one will be
                    interpolated. If no exp_time is given (not recommended!), all dark files in 'dark'
                    will simply be combined using combine_mode.
    :param save_masters: option to save master files. Default is false. Saves in raw AstroDir path.
    :param save_reduced: option to save reduced files. Default is false. Saves in sci AstroDir path.
    :return: AstroDir of reduced science images and corresponding master files attached.
    """

    if bias is not None:
        #bias_data = bias.readdata()
        warnings.warn('Combining bias with function: %s' % (combine_mode))
        mb = get_masterbias(bias, combine_mode, save_masters)
    else:
        mb = raw.bias
    if dark is not None:
        #dark_data = dark.readdata()
        warnings.warn('Combining darks with function: %s' % (combine_mode))
        md = get_masterdark(dark, combine_mode, exp_time, save_masters)
    else:
        md = raw.dark
    if flat is not None:
        #flat_data = flat.readdata()
        warnings.warn('Combining flats with function: %s' % (combine_mode))
        mf = get_masterflat(flat, combine_mode, save_masters)
    else:
        mf = raw.flat

    raw_data = raw.readdata()
    sci = []

    import pyfits as pf
    i = 0

    for r in raw_data:
        s = (r - mb - (md - mb)) / (mf - mb)
        #s_header = r.readheader()  # Reduced data header is same as raw header for now
        hdu = pf.PrimaryHDU(s)
        filename = sci_path + "/reduced_" + "%03i.fits" % i
        hdu.writeto(filename)
        i += 1

    return io_dir.AstroDir(sci_path, mb, mf, md)


def GPUreduce(bias, dark, flat, raw, combine_mode=CPUmath.mean_combine, save_masters=False):
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

    warnings.warn('Using combine function: %s' % (combine_mode))

    mb = get_masterbias(bias, combine_mode, save_masters)
    md = get_masterdark(dark, combine_mode, save_masters)
    mf = get_masterflat(flat, combine_mode, save_masters)

    img = np.array([])

    for fi in raw:
        fi = fi - mb
        sh = fi.shape
        ss = sh[0] * sh[1]
        data = fi.reshape(1, ss)
        ndata = data[0]
        img = np.append(img, ndata)

    # GPU reduction
    img_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    dark_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=md - mb)
    flat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mf - mb)
    res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)

    f = open('reduce.cl', 'r')
    programName = "".join(f.readlines())

    program = cl.Program(ctx, programName).build()
    program.reduce(queue, img.shape, None, dark_buf, flat_buf, img_buf, res_buf)  # sizeX, sizeY, sizeZ

    res = np.empty_like(img)
    cl.enqueue_copy(queue, res, res_buf)
