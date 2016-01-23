from src import TimeSeries
import numpy as np

a = np.arange(20)
b = np.arange(10, 30)
c = np.arange(20, 40)

a_e = np.arange(20)*0.1
b_e = np.arange(10, 30)*0.01
c_e = np.arange(20, 40)*0.1

labels = ['target', 'ref1', 'ref2']

t1 = TimeSeries.TimeSeries([a, b, c], [a_e, b_e, c_e], labels)