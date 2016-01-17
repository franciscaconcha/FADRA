__author__ = 'fran'

import CPUmath
import pyfits as pf
import warnings
from functools import wraps as _wraps
import numpy as np

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
                    warnings.warn("File %s is of different shape. Will be ignored in %s." % (i, function.__name__) )
                    #raise IOError("Files to be combined in CPUmath.%s are not all of the same shape." % function.__name__)
                else:
                    raise IOError("Files to be combined in CPUmath.%s are not all of the same type." % function.__name__)
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

#Interpolates darks to necessary exposure time
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
def get_masterdark(darks, combine_mode, save, time=None):
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
        master_dark = interpol(darks, time)

    if save:
        hdu = pf.PrimaryHDU(master_dark)
        hdu.writeto('MasterDark.fits')

    print("MasterDark done")
    return master_dark

def CPUreduce(bias, dark, flat, raw, combine_mode=CPUmath.mean_combine, save_masters=False):
    """ Reduces image files. Combines bias, flat, and dark files to obtain masters.
        Combination function given in combine_mode, default is CPU mean_combine.
        Returns array of reduced sci images. Can save masters if save_masters = True.
    :param bias: np.ndarray of all bias files to be combined
    :param flat: np.ndarray of all flat files to be combined
    :param dark: np.ndarray of all dark files to be combined
    :param raw: np.ndarray of all raw files to be reduced
    :param combine_mode: function used to combine bias, darks, and flats
    :param save_masters: option to save master files. Default is false. Saves in current folder.
    :return: np.ndarray of reduced science images
    """
    warnings.warn('Using combine function: %s' % (combine_mode))

    mb = get_masterbias(bias, combine_mode, save_masters)
    md = get_masterdark(dark, combine_mode, save_masters)
    mf = get_masterflat(flat, combine_mode, save_masters)

    sci = []

    for r in raw:
        s = (r - mb - (md - mb))/(mf - mb)
        sci.append(s)

    return sci


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

    #GPU reduction
    img_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    dark_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=md-mb)
    flat_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mf-mb)
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