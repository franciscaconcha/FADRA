from src import get_stamps
import dataproc as dp
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import time

basepath = "/media/Fran/data/"

folders = [{"path": "sara/20131214/", "targets": [[520, 365], [434, 625]], "ap": 20, "sky": [25, 30],
            "stamp": 50, "bias": "Bias-011.fits", "dark": "Dark30s-000.fits", "flat": "Flat5-001.fits"},
           {"path": "sara/20140105/", "targets": [[498, 465], [417, 722]], "ap": 20, "sky": [25, 30],
            "stamp": 50, "bias": "bias-000.fits", "dark": "dark200s-000.fits", "flat": "flats1s000.fits"},
           {"path": "sara/20140118/", "targets": [[570, 269], [436, 539]], "ap": 12, "sky": [16, 20],
            "stamp": 30, "bias": "bias-000.fits", "dark": "dark200s-000.fits", "flat": "flats1s004.fits"}
           ]

curr_time = time.strftime("%d-%m-%Y-%H-%M-%S")
fp = open('./results/time_lightcurves-' + curr_time + '.txt', 'w+')
fp.write("Dataset \t CPU \t GPU \t NRMSE\n")

for f in folders:
    f_path = basepath + f['path']
    io = dp.AstroDir(f_path + 'sci/')
    dark = dp.AstroFile(f_path + 'dark/' + f['dark'])
    bias = dp.AstroFile(f_path + 'bias/' + f['bias'])
    flat = dp.AstroFile(f_path + 'flat/' + f['flat'])
    res_gpu, gpu_time = get_stamps.photometry(io, bias, dark, flat,
                                   f['targets'], f['ap'], f['stamp'], f['sky'],
                                   gain=1.0, ron=None, gpu=True)
    res_cpu, cpu_time = get_stamps.photometry(io, bias, dark, flat,
                                   f['targets'], f['ap'], f['stamp'], f['sky'],
                                   gain=1.0, ron=None)

    #res_cpu.plot()
    #res_gpu.plot()

    rc = [int(i) for i in res_cpu[0]]

    rmse = np.array(sqrt(mean_squared_error(rc, res_gpu[0])))
    nrmse = rmse/(max(np.array(res_cpu[0])) - min(np.array(res_gpu[0])))

    fp.write("%s \t %.2f s \t %.2f s \t %.4f\n" % (f['path'], cpu_time, gpu_time, nrmse))
    #print("%s || CPU: %.2f s || GPU: %.2f s || NRMSE: %.4f\n" % (f['path'], cpu_time, gpu_time, nrmse))

fp.close()
