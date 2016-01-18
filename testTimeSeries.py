from src import TimeSeries
import numpy as np

a = np.arange(20)
b = np.arange(10, 30)
c = np.arange(20, 40)

a_e = range(20)
b_e = range(10, 30)
c_e = range(20, 40)

labels = ['target', 'ref1', 'ref2']

t1 = TimeSeries.TimeSeries([a, b, c], [a_e, b_e, c_e], labels)