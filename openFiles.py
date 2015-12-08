__author__ = 'fran'

import pyfits as pf
import warnings
from functools import wraps as _wraps

def get_file_list(fpath):
    """
    Receives a path and returns a list of all the files within that
    path. The names are returned with the path included, ready to
    be open. Subfolders are ignored.
    :param fpath: path to files folder
    :return: [filenames]
    """
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [fpath + "/" + f for f in listdir(fpath) if isfile(join(fpath, f))]
    return onlyfiles


def open_files(filelist):
    """
    Receives list of filenames. Returns 2 lists: one with
    science data and one with the files' headers.
    :param filelist: list of filenames to open
    :return: [list, list] [data, headers]
    """
    data = []
    headers = []
    for f in filelist:
        data.append(pf.getdata(f))
        headers.append(pf.getheader(f))
    return [data, headers]


def separate_paths(biaspath, darkpath, flatpath, rawpath):
    """
    Handles file opening when paths are given separately
    for each kind of file. If only one file is wanted then
    it must be the sole file in the given path.
    :param biaspath: path to BIAS files folder
    :param darkpath: path to Dark files folder
    :param flatpath: path to Flat files folder
    :param rawpath: path to raw science images folder
    :return: [[biasdata, biasheaders], [darkdata, darkheaders], [flatdata, flatheaders], [rawdata, rawheaders]]
    """
    print("Opening BIAS files...")
    bias_files = get_file_list(biaspath)
    bias = open_files(bias_files)
    print("Done.")

    print("Opening Dark files...")
    dark_files = get_file_list(darkpath)
    darks = open_files(dark_files)
    print("Done.")

    print("Opening Flat files...")
    flat_files = get_file_list(flatpath)
    flats = open_files(flat_files)
    print("Done.")

    print("Opening Raw Science files...")
    raw_files = get_file_list(rawpath)
    raws = open_files(raw_files)
    print("Done.")

    return [bias, darks, flats, raws]

def one_path(fpath):
    """
    Handles file opening when all BIAS, dark, flat, and raw
    science files are within one common folder. Will use header
     data of each file to separate between categories.
     USE AT YOUR OWN RISK! Make sure all headers have the right info first!
    :param fpath: path to folder with all the files
    :return: [[biasdata, biasheaders], [darkdata, darkheaders], [flatdata, flatheaders], [rawdata, rawheaders]]
    """
    warnings.warn('Given path: %s \nWill try to separate all files in path between'
                  'bias, darks, flats, and raw science image. \nWill only use header'
                  'info for this. \nResults may NOT be accurate. \nMake sure you have'
                  'checked your files\' headers!' % (fpath))

    biasdata, biasheaders = [], []
    darkdata, darkheaders = [], []
    flatdata, flatheaders = [], []
    rawdata, rawheaders = [], []

    files_path = get_file_list(fpath)
    allfiles = open_files(files_path)

    for f in allfiles:
        file_header = pf.getheader(f)
        file_type = file_header['OBJECT']
        if file_type == 'BIAS':
            biasdata.append(pf.getdata(f))
            biasheaders.append(file_header)
        elif file_type[:4] == 'FLAT':
            flatdata.append(pf.getdata(f))
            flatheaders.append(file_header)
        elif file_type == 'DARK':
            darkdata.append(pf.getdata(f))
            darkheaders.append(file_header)
        else:
            rawdata.append(pf.getdata(f))
            rawheaders.append(file_header)

    return [[biasdata, biasheaders], [darkdata, darkheaders], [flatdata, flatheaders], [rawdata, rawheaders]]


