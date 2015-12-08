__author__ = 'fran'

from src import reduction, CPUmath
import numpy as np
import sys
import openFiles

def gen(x, y, n):
    r = []
    for i in range(n):
        rr = np.random.randint(500, size=(x, y))
        r.append(rr)
    return r

def main():
    try:
        x = sys.argv[1] #path bias
        y = sys.argv[2] #path dark
        n = sys.argv[3] #path flat
        z = sys.argv[4] #path raw
        s = sys.argv[5] #path science
    except:
        print('Unexpected error: ', sys.exc_info()[0])
        raise

    """b = gen(x, y, n)
    d = gen(x, y, n + 1)
    f = gen(x, y, n + 2)
    r = gen(x, y, n)"""

    b, d, f, r = openFiles.separate_paths(x, y, n, z)

    print(len(b), len(b[0]), len(b[1]))
    print(len(f), len(f[0]), len(f[1]))
    print(len(d), len(d[0]), len(d[1]))
    print(len(r), len(r[0]), len(r[1]))

    print(b[0][0].shape, b[0][1].shape)

    print("Starting reduction...")
    s = reduction.reduce(b[0], d[0], f[0], r[0], combine_mode=CPUmath.median_combine)
    print("Done!")

    print(len(s))

if __name__ == "__main__":
    main()
