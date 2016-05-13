import ccdproc
from astropy import units
import time
from glob import glob
import dataproc as dp
from fractal.src import reduction #import GPUreduce, CPUreduce
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

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
fp = open('./results/time_reduction/' + curr_time + '.txt', 'w+')
fp.write("Dataset \t CPU_AP \t CPU_F \t GPU \t NRMSE_CPU \t NMAE_CPU \t NRMSE_GPU \t NMAE_GPU \n")

for f in folders:
    f_path = basepath + f['path']

    # astropy
    io_ap = f_path + 'sci10/*'
    dark_ap = f_path + 'dark/' + f['dark']
    bias_ap = f_path + 'bias/' + f['bias']
    flat_ap = f_path + 'flat/' + f['flat']

    dirs = glob(io_ap)

    bias = ccdproc.CCDData.read(bias_ap, unit="electron")
    dark = ccdproc.CCDData.read(dark_ap, unit="electron")
    flat = ccdproc.CCDData.read(flat_ap, unit="electron")

    cpu_reduced = []
    gpu_reduced = []

    #ic = ccdproc.ImageFileCollection(dirs, keywords="*")
    t0 = time.clock()

    for d in dirs: #ic.data():
        data = ccdproc.CCDData.read(d, unit="electron")
        data_no_bias = ccdproc.subtract_dark(data, bias, data_exposure=20*units.second,
                                             dark_exposure=0*units.second)
        dark_no_bias = ccdproc.subtract_dark(dark, bias, data_exposure=20*units.second,
                                             dark_exposure=0*units.second)
        data_no_dark = ccdproc.subtract_dark(data_no_bias, dark_no_bias, data_exposure=20*units.second,
                                             dark_exposure=0*units.second)
        flat_no_bias = ccdproc.subtract_dark(flat, bias, data_exposure=20*units.second,
                                             dark_exposure=0*units.second)
        ima_reduced = ccdproc.flat_correct(data_no_dark, flat_no_bias)
        cpu_reduced.append(ima_reduced)

    cpu_time = time.clock() - t0

    # FADRA
    io_f = dp.AstroDir(f_path + 'sci10/')
    dark_f = dp.AstroFile(f_path + 'dark/' + f['dark'])
    bias_f = dp.AstroFile(f_path + 'bias/' + f['bias'])
    flat_f = dp.AstroFile(f_path + 'flat/' + f['flat'])

    res_cpu, cpu_f_time = reduction.reduce(io_f, f_path + 'reduced/', bias_f, dark_f, flat_f)
    print("CPU reduce done")
    res_gpu, gpu_f_time = reduction.reduce(io_f, f_path + 'reduced/', bias_f, dark_f, flat_f, gpu=True)

    all_nrmse = []
    all_nrmse_g = []
    all_nmae = []
    all_nmae_g = []

    all_mean = []
    all_mean_g = []

    for i in range(len(res_cpu)):
        cpu_fadra = res_cpu[i]
        cpu_ap = cpu_reduced[i]

        rmse = np.array(sqrt(mean_squared_error(cpu_fadra, np.asarray(cpu_ap))))
        nrmse = rmse/(np.amax(np.array(cpu_fadra)) - np.amin(np.asarray(cpu_ap)))
        #all_nrmse.append(nrmse)
        all_nrmse.append(np.mean(cpu_fadra))

        mae = np.sum(np.absolute(cpu_fadra - np.asarray(cpu_ap)))
        nmae = mae/(np.amax(np.array(cpu_fadra)) - np.amin(np.asarray(cpu_ap)))
        #all_nmae.append(nmae)
        all_nmae.append(np.median(np.asarray(cpu_ap)))

        all_mean.append(np.mean(cpu_fadra - np.asarray(cpu_ap)))

        cpu_ap = res_gpu[i]

        rmse_g = np.array(sqrt(mean_squared_error(cpu_fadra, np.asarray(cpu_ap))))
        nrmse_g = rmse_g/(np.amax(np.array(cpu_fadra)) - np.amin(np.array(cpu_ap)))
        #all_nrmse_g.append(nrmse_g)
        all_nrmse_g.append(np.mean(cpu_fadra) - np.mean(cpu_ap))

        mae_g = np.sum(np.absolute(cpu_fadra - np.asarray(cpu_ap)))
        nmae_g = mae_g/(np.amax(np.array(cpu_fadra)) - np.amin(np.array(cpu_ap)))
        all_nmae_g.append(nmae_g)

        all_mean_g.append(np.mean(cpu_fadra - cpu_ap))

    #fp.write("%s \t %.2f s \t %.2f s  \t %.2f s \t %.4f \t %.4f \t %.4f \t %.4f\n" %
#             (f['path'], cpu_time, cpu_f_time, gpu_f_time, np.mean(all_nrmse), np.mean(all_nmae),
#              np.mean(all_nrmse_g), np.mean(all_nmae_g)))
    fp.write("%s \t %.2f s \t %.2f s  \t %.2f s \t %.4f \t %.4f\n" % (f['path'], cpu_time, cpu_f_time, gpu_f_time,
             np.mean(all_mean)/(max(all_mean) - min(all_mean)),
             np.mean(all_mean_g)/(max(all_mean_g) - min(all_mean_g))))

fp.close()