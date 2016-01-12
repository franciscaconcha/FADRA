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
    print(limit)
    return [limit] + list(np.random.randint(limit/2, size=((n-1),)))


def main():
    config = configparser.ConfigParser()
    config.read("config")
    values = config['DEFAULT']
    size = int(values['Image_size'].split()[0])
    n = int(values['N_of_stars'].split()[0])
    margin = int(values['Margin'].split()[0])
    max_exp_t = int(values['Max_exp_time'].split()[0])

    coords = createCoords(n, size, margin)
    rads = createRads(n, 20)
    img = np.zeros((size, size))

    i = 0

    for c in coords:
        img += makeGaussian(size, rads[i], c)
        i += 1

    plt.matshow(img)
    plt.show()

if __name__ == "__main__":
    main()
