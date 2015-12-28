__author__ = 'fran'

import scipy as sp
import warnings

class TimeSeries(object):

    def __init__(self, data, errors, ids):
        self.channels = data  # [[target1_im1, target1_im2, ...], [target2_im1, target2_im2, ...]]
        self.errors = errors
        # [[error_target1_im1, error_target1_im2, ...], [error_target2_im1, error_target2_im2, ...]]
        self.group = [1] + [0 for i in range(len(data)-1)]
        # Default grouping: 1st coordinate is 1 group, all other objects are another group
        self.ids = ids  # Dictionary for names?

        self.channels.append([])  # Group 1 operation result; is overwritten every time a new op is defined
        self.errors.append([])

        self.channels.append([])  # Group 2 operation result; is overwritten every time a new op is defined
        self.errors.append([])

    def __getitem__(self, item):
        # This is so I can do ts[0]/ts[2] and it works directly with the channels!
        return self.channels[item]

    def group1(self):
        return [self.channels[i] for i in range(len(self.channels)) if self.group[i]]

    def group2(self):
        return [self.channels[i] for i in range(len(self.channels)) if not self.group[i]]

    def set_group(self, new_group):  # Receives pe [0 1 0 0 1 0] and that is used to define 2 groups
        self.group = new_group

    def mean(self, group_id):
        if group_id > 2:
            warnings.warn("group_id must be 1 or 2 only. Group 2 will be used as default.")
            group = self.group2()
        elif group_id == 1:
            group = self.group1()
        else:
            group = self.group2()

        self.channels[-group_id] = sp.mean(group, axis=0)

    def median(self, group_id):
        if group_id > 2:
            warnings.warn("group_id must be 1 or 2 only. Group 2 will be used as default.")
            group = self.group2()
        elif group_id == 1:
            group = self.group1()
        else:
            group = self.group2()

        self.channels[-group_id] = sp.median(group, axis=0)