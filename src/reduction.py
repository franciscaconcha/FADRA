__author__ = 'fran'

import CPUmath
import pyfits as pf
import warnings
from functools import wraps as _wraps

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
                    raise IOError("Files to be combined in CPUmath.%s are not all of the same shape." % function.__name__)
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

    return master_flat

@_check_combine_input
def get_masterdark(darks, combine_mode, save):
    """ Returns masterdark, combining all dark files using the given function.
        If not, uses CPU mean as default. Returns masterdark file and brings
        option to save it to .fits file.
    :param flats: np.ndarray with all dark arrays
    :param combine_mode: function used to combine darks
    :param save: true if want to save the master dark file. Default is false.
    :return: np.ndarray (master dark)
    """
    master_dark = combine_mode(darks)

    if save:
        hdu = pf.PrimaryHDU(master_dark)
        hdu.writeto('MasterDark.fits')

    return master_dark

def reduce(bias, flat, dark, raw, combine_mode=CPUmath.mean_combine, save_masters=False):
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
    mf = get_masterbias(flat, combine_mode, save_masters)
    md = get_masterbias(dark, combine_mode, save_masters)

    sci = []

    for r in raw:
        s = (r - mb - md)/mf
        sci.append(s)

    return sci
