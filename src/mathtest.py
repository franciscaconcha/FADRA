__author__ = 'fran'

import numpy as np
import pyfits as pf
import CPUmath

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[3, 4, 5], [6, 7, 8]])
c = np.array([[9, 10, 11], [12, 1, 2]])
#print(CPUmath.median_combine([a, b, c]))
#print(CPUmath.mean_combine([a, b, c]))

#hdu_list = pf.open('test.fits')
#print(type(hdu_list), type(a))

#header = hdu_list[0].header
#sci_data = hdu_list[0].data
#print(type(header), type(sci_data))

#CPUmath.mean_combine(a, b, c, np.array([1,2]))

from reduction import Reduction

r = Reduction()
r._load_data('/media/Fran/dataproc/test')
