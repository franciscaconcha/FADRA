import dataproc as dp
import numpy as np

io = dp.AstroDir("/media/Fran/2011_rem/rawsci70/raw6/")
io_iraf = dp.AstroDir("/media/Fran/2011_rem/rawsci70/raw6/raw6-2/")

my_data = io.readdata()
iraf_data = io_iraf.readdata()

from fractal.src.CPUmath import mean_filter, median_filter, shift_generator

mine_filtered = []

"""import matplotlib.pyplot as plt
fig = plt.figure()
d = shift_generator(my_data[0], 100, 'constant')
for i in d:
    plt.imshow(i)
    plt.show()"""


for i in my_data:
    mine_filtered.append(mean_filter(i, 2))

from astropy.convolution import convolve, Box2DKernel
astro_filtered = []
box_2D_kernel = Box2DKernel(2)
for i in my_data:
    astro_filtered.append(convolve(i, box_2D_kernel, boundary='extend'))

for i, j in zip(mine_filtered, astro_filtered):
    dif = i - j
    rms = np.sqrt(np.mean(np.square(dif)))
    print rms