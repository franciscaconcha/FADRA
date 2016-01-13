import numpy as np
import sys
import matplotlib.pyplot as plt
import configparser


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def createCoords(n, size, margin):
    c = []
    for i in range(n):
        c.append([np.random.randint(margin, size-margin), np.random.randint(margin, size-margin)])
    return c


def createRads(n, limit):
    return [limit] + list(np.random.randint(limit/2, size=((n-1),)))


def main():
    # Read configuration file
    config = configparser.ConfigParser()
    config.read("config")

    field = config['FIELD']
    size = int(field['Image_size'].split()[0])
    margin = int(field['Margin'].split()[0])

    stars = config['STARS']
    n = int(stars['N_of_stars'].split()[0])
    fwhm = int(stars['FWHM'].split()[0])

    dark = config['DARK']
    max_exp_t = int(dark['Max_exp_time'].split()[0])
    max_noise = float(dark['Max_noise_level'].split()[0])

    flat = config['FLAT']
    sky = flat['Sky']

    dinfo = config['DATA']
    n_imgs = int(dinfo['N_of_images'].split()[0])
    repeat_stars = dinfo['Repeat_stars'].split()[0]
    save_clean = bool(dinfo['Save_clean'].split()[0])
    save_noisy = bool(dinfo['Save_noisy'].split()[0])
    save_masters = bool(dinfo['Save_masters'].split()[0])
    save_reduced = bool(dinfo['Save_reduced'].split()[0])

    # Generate darks going from 1 s to max_exp_time
    darks = []
    for i in range(1, max_exp_t + 1):
        darks.append(np.random.rand(size, size) * max_noise * fwhm * i)

    # Generate flat

    # Generate coordinates and stars
    images = []

    if repeat_stars == "True":
        coords = createCoords(n, size, margin)
        rads = createRads(n, 20)
        img = np.zeros((size, size))

        j = 0

        for c in coords:
            img += makeGaussian(size, rads[j], c)
            j += 1

        plt.matshow(img)
        plt.show()

        images = [img for k in range(n_imgs)]

    else:
        for i in range(n_imgs):
            coords = createCoords(n, size, margin)
            rads = createRads(n, 20)
            img = np.zeros((size, size))

            j = 0

            for c in coords:
                img += makeGaussian(size, rads[j], c)
                j += 1

            images.append(img)

    for p in images:
        plt.matshow(p)
        plt.show()

if __name__ == "__main__":
    main()
