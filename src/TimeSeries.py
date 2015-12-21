__author__ = 'fran'

class TimeSeries(object):

    def __init__(self, data, errors, ids):
        self.channels = data
        self.errors = errors
        self.group = [1] + [0 for i in range(len(data)-1)]
        self.ids = ids

    def group1(self):
        return [self.channels[i] for i in range(len(self.channels)) if self.group[i]]

    def group2(self):
        return [self.channels[i] for i in range(len(self.channels)) if not self.group[i]]

    def set_group(self, new_group): # Receives pe [0 1 0 0 1 0] and that is used to define 2 groups
        self.group = new_group

    def average(self):
