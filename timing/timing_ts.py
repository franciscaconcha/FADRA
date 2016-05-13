from src import get_stamps
import dataproc as dp
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import time

basepath = "/media/Fran/data/"

folders = [{"path": "sara/20110824/", "targets": [[484, 439], [923, 844]], "ap": 10, "sky": [15, 20],
            "stamp": 30, "bias": "Bias-20110824-001.FITS", "dark": "Dark-20110824-001.FITS",
            "flat": "flatI001.fits"},
           {"path": "sara/20120430/", "targets": [[830, 680], [554, 1164]], "ap": 20, "sky": [25, 30],
            "stamp": 40, "bias": "bias-000.fits", "dark": "dark-000.fits", "flat": "flat-001.fits"},
           {"path": "sara/20140118/", "targets": [[570, 269], [436, 539]], "ap": 12, "sky": [16, 20],
            "stamp": 30, "bias": "bias-000.fits", "dark": "dark200s-000.fits", "flat": "flats1s004.fits"},
           {"path": "sara/20130422/", "targets": [[677, 653], [462, 625]], "ap": 9, "sky": [12, 16],
            "stamp": 20, "bias": "Dark-20110824-001.FITS", "dark": "Dark-20110824-001.FITS",
            "flat": "flatI001.fits"},
           {"path": "sara/20131117/", "targets": [[741, 684], [520, 286]], "ap": 5, "sky": [7, 9],
            "stamp": 15, "bias": "calib-20131118-None--bias-654.fits", "dark": "calib-20131118-None-604.fits",
            "flat": "flats-20131117-Bessell R-001.fits"},
           {"path": "sara/20131214/", "targets": [[520, 365], [434, 625]], "ap": 20, "sky": [25, 30],
            "stamp": 50, "bias": "Bias-011.fits", "dark": "Dark30s-000.fits", "flat": "Flat5-001.fits"},
           {"path": "sara/20131108/", "targets": [[1167, 920], [1291, 1210]], "ap": 10, "sky": [12, 15],
            "stamp": 20, "bias": "DarksNone001.fits", "dark": "DarksNone001.fits",
            "flat": "Flat10Bessell R001.fits"},
           {"path": "sara/20140105/", "targets": [[498, 465], [417, 722]], "ap": 20, "sky": [25, 30],
            "stamp": 50, "bias": "bias-000.fits", "dark": "dark200s-000.fits", "flat": "flats1s000.fits"}
           ]

curr_time = time.strftime("%d-%m-%Y-%H-%M-%S")
fp = open('./results/time_lightcurves/' + curr_time + '.txt', 'w+')
fp.write("Dataset \t CPU \t GPU \t NRMSE \t NMAE\n")

for f in folders:
    f_path = basepath + f['path']
    print(f_path)
    io = dp.AstroDir(f_path + 'sci10/')
    dark = dp.AstroFile(f_path + 'dark/' + f['dark'])
    bias = dp.AstroFile(f_path + 'bias/' + f['bias'])
    flat = dp.AstroFile(f_path + 'flat/' + f['flat'])
    res_gpu, gpu_time = get_stamps.photometry(io, bias, dark, flat,
                                   f['targets'], f['ap'], f['stamp'], f['sky'],
                                   gain=1.0, ron=5., gpu=True)
    res_cpu, cpu_time = get_stamps.photometry(io, bias, dark, flat,
                                   f['targets'], f['ap'], f['stamp'], f['sky'],
                                   gain=1.0, ron=5.)

    #from fractal.src import TimeSeries
    #ts1 = TimeSeries.TimeSeries([res_cpu.channels[0], res_gpu.channels[0]], [res_cpu.errors[0], res_gpu.errors[0]],
    #                    labels=['TARGET', 'REF1'], epoch=res_cpu.epoch)

    #ts1.plot()

    #res_cpu.plot()
    #res_gpu.plot()

    print(np.array(res_gpu[0])/np.array(res_cpu[0]))

    rc = [int(i) for i in res_cpu[0]]

    rmse = np.array(sqrt(mean_squared_error(rc, res_gpu[0])))
    nrmse = rmse/(max(np.array(res_gpu[0])) - min(np.array(res_gpu[0])))

    mae = np.sum(np.absolute(np.array(rc) - np.array(res_gpu[0])))
    nmae = mae/(max(np.array(res_gpu[0])) - min(np.array(res_gpu[0])))

    fp.write("%s \t %.2f s \t %.2f s \t %.4f \t %.4f\n" % (f['path'], cpu_time, gpu_time, nrmse, nmae))
    #print("%s || CPU: %.2f s || GPU: %.2f s || NRMSE: %.4f\n" % (f['path'], cpu_time, gpu_time, nrmse))

fp.close()
