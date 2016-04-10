import time
from astropy.convolution import convolve, Box2DKernel
from fractal.src.CPUmath import mean_filter, median_filter
import dataproc as dp

path = "/media/Fran/2011_rem/rawsci70/raw6/"

io = dp.AstroDir("/media/Fran/2011_rem/rawsci70/raw6/")
io_data = io.readdata()

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

# PyRAF filtering
"""from pyraf import iraf
t0 = time.clock()
iraf.images()
j = 0
for i in onlyfiles:
    iraf.imfilter.boxcar(path + i, path + "test", 5, 5)
    print j
    j += 1
time_pyraf = time.clock() - t0
print time.strftime("%M:%S +0000", time_pyraf)"""

# AstroPy filtering
t0 = time.clock()
box_2D_kernel = Box2DKernel(5)
for i in io_data:
    convolve(i, box_2D_kernel, boundary='extend')
time_astropy = time.clock() - t0

# FADRA filtering
t0 = time.clock()
for i in io_data:
    mean_filter(i, 5)
time_fadra = time.clock() - t0

print("AstroPy: %.2f || FADRA: %.2f" % (time_astropy, time_fadra))


