import dataproc as dp
from src import reduction
import time
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def nrmse(data1, data2):
    rmse = np.array(sqrt(mean_squared_error(data1, data2)))
    return rmse/np.array(sqrt(mean_squared_error(data1, np.zeros(data1.shape))))

basepath = "/media/Fran/data/"

folders = [#{"path": "sara/20130422/", "targets": [[520, 365], [434, 625]], "ap": 20, "sky": [25, 30],
           # "stamp": 50, "bias": "DarkffNone001.fits", "dark": "DarkffNone001.fits", "flat": "flatI001.fits"},
           #{"path": "sara/20131108/", "targets": [[520, 365], [434, 625]], "ap": 20, "sky": [25, 30],
           # "stamp": 50, "bias": "BiasNone001.fits", "dark": "DarkffNone001.fits", "flat": "Flat10Bessell R001.fits"},
           #{"path": "sara/20131117/", "targets": [[520, 365], [434, 625]], "ap": 20, "sky": [25, 30],
           # "stamp": 50, "bias": "calib-20131118-None--bias-654.fits", "dark": "calib-20131118-None-604.fits",
           # "flat": "flats-20131117-Bessell R-001.fits"},
           {"path": "sara/20131214/", "targets": [[520, 365], [434, 625]], "ap": 20, "sky": [25, 30],
            "stamp": 50, "bias": "Bias-011.fits", "dark": "Dark30s-000.fits", "flat": "Flat5-001.fits"},
           {"path": "sara/20140105/", "targets": [[498, 465], [417, 722]], "ap": 20, "sky": [25, 30],
            "stamp": 50, "bias": "bias-000.fits", "dark": "dark200s-000.fits", "flat": "flats1s000.fits"},
           {"path": "sara/20140118/", "targets": [[570, 269], [436, 539]], "ap": 12, "sky": [16, 20],
            "stamp": 30, "bias": "bias-000.fits", "dark": "dark200s-000.fits", "flat": "flats1s004.fits"}
           ]

curr_time = time.strftime("%d-%m-%Y-%H-%M-%S")
fp = open('./results/time_reduction/' + curr_time + '.txt', 'w+')
fp.write("Dataset \t CPU \t GPU\n")

for f in folders:
    f_path = basepath + f['path']
    reduced_path = f_path + 'reduced/'
    io = dp.AstroDir(f_path + 'sci1/')
    dark = dp.AstroFile(f_path + 'dark/' + f['dark'])#.reader()
    bias = dp.AstroFile(f_path + 'bias/' + f['bias'])#.reader()
    flat = dp.AstroFile(f_path + 'flat/' + f['flat'])#.reader()

    for the_file in os.listdir(reduced_path):
        file_path = os.path.join(reduced_path, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)

    res_gpu, gpu_time = reduction.reduce(io, reduced_path, bias, dark, flat, gpu=True)
    print("GPU reduction done")
    res_cpu, cpu_time = reduction.reduce(io, reduced_path, bias, dark, flat)
    print("CPU reduction done")

    fp.write("%s \t %.2f s \t %.2f s\n" % (f['path'], cpu_time, gpu_time))

fp.close()

#print len(res_cpu)
#diff = [np.isnan(res_cpu[i] - res_gpu[i]).any() for i in range(len(res_cpu))]
#print diff

m_cpu = [np.mean(i) for i in res_cpu]
m_gpu = [np.mean(i) for i in res_gpu]

print m_cpu
print m_gpu

#for i in res_gpu:
#    print np.isinf(i).any()

#from scipy.stats import nanmean, nanstd
#print nanmean(diff), nanstd(diff)


