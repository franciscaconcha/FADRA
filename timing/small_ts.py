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

for f in folders:
    curr_time = time.strftime("%d-%m-%Y-%H-%M-%S")
    fp = open('./results/small_ts/' + f['path'][5:-1] + '-' + curr_time + '.txt', 'w+')
    fp.write("Epoch \t CPU \t GPU\n")

    f_path = basepath + f['path']
    print(f_path)
    io = dp.AstroDir(f_path + 'sci10/')
    dark = dp.AstroFile(f_path + 'dark/' + f['dark'])
    bias = dp.AstroFile(f_path + 'bias/' + f['bias'])
    flat = dp.AstroFile(f_path + 'flat/' + f['flat'])
    res_gpu, gpu_time = get_stamps.photometry(io, bias, dark, flat,
                                   f['targets'], f['ap'], f['stamp'], f['sky'],
                                   gain=1.0, ron=1.0, gpu=True)
    res_cpu, cpu_time = get_stamps.photometry(io, bias, dark, flat,
                                   f['targets'], f['ap'], f['stamp'], f['sky'],
                                   gain=1.0, ron=1.0)

    for i in range(len(res_cpu[0])):
        fp.write("%s \t %d \t %d\n" % (res_cpu.epoch[i], res_cpu[0][i], res_gpu[0][i]))

    #res_cpu.plot()
    #res_gpu.plot()

fp.close()
