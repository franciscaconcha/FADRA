__author__ = 'fran'

from src import reduction, CPUmath
import numpy as np
import sys

def gen(x, y, n):
    r = []
    for i in range(n):
        rr = np.random.randint(500, size=(x, y))
        r.append(rr)
    return r

def main():
    try:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
        n = int(sys.argv[3])
    except:
        print('Unexpected error: ', sys.exc_info()[0])
        raise

    b = gen(x, y, n)
    d = gen(x, y, n + 1)
    f = gen(x, y, n + 2)
    r = gen(x, y, n)

    s = reduction.reduce(b, f, d, r, combine_mode=CPUmath.median_combine)

if __name__ == "__main__":
    main()
