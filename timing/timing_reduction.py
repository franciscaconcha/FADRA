import pyfits as pf
import dataproc as dp
from src import reduction

path = '/media/Fran/2013-03-31/'
sci_path = path + 'results'
dirs = ['sci5']#, 'sci10', 'sci50', 'sci100', 'sci500', 'sci1000']

mbias = pf.getdata(path + 'bias.fits')
mflat = pf.getdata(path + 'flat.fits')
mdark = pf.getdata(path + 'bias.fits')

import os
for the_file in os.listdir(sci_path):
    file_path = os.path.join(sci_path, the_file)
    if os.path.isfile(file_path):
        os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)

for the_file in os.listdir('/media/Fran/fractal/timing/results/'):
    file_path = os.path.join('/media/Fran/fractal/timing/results/', the_file)
    if os.path.isfile(file_path):
        os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)

f = open('/media/Fran/fractal/timing/results/time_reduction.dat', 'w')
f.write('N \t CPU \t GPU\n')

for d in dirs:
    io = dp.AstroDir(path + d)
    r_gpu, t_gpu = reduction.reduce(io, sci_path, [mbias], [mdark], [mflat], gpu=True)
    print("GPU done")
    r_cpu, t_cpu = reduction.reduce(io, sci_path, [mbias], [mdark], [mflat])
    print("CPU done")
    line = '%s \t %.2f \t %.2f\n' % (d, t_cpu, t_gpu)
    f.write(line)

f.close()
