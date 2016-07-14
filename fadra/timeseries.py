__author__ = "Francisca Concha"
__license__ = "GPLv3"
__email__ = "faconcha@dcc.uchile.cl"
__status__ = "Development"

import scipy as sp
import warnings

class TimeSeries(object):

    def __init__(self, data, errors, labels=None, epoch=None):
        dc = []
        for d in data:
            dc.append(sp.array(d))  # [[target1_im1, target1_im2, ...], [target2_im1, target2_im2, ...]]
        self.channels = dc
        de = []
        for e in errors:
            de.append(sp.array(e))
        self.errors = de
        # [[error_target1_im1, error_target1_im2, ...], [error_target2_im1, error_target2_im2, ...]]
        self.group = [1] + [0 for i in range(len(data)-1)]
        # Default grouping: 1st coordinate is 1 group, all other objects are another group
        self.labels = {}
        if labels is not None:
            self.labels = self.set_labels(labels)  # Dictionary for names?

        self.channels.append([])  # Group 1 operation result; is overwritten every time a new op is defined
        self.errors.append([])

        self.channels.append([])  # Group 2 operation result; is overwritten every time a new op is defined
        self.errors.append([])

        self.epoch = epoch

    # This is so I can do ts[0]/ts[2] and it works directly with the channels
    def __getitem__(self, item, error=False):
        """ Returns the corresponding TimeSeries channel, which corresponds to the photometry
            result of the target given in item.
        :param item: ID of the target. Can be number or label name.
        :param error: if True, returns the error channel instead of the photometry channel.
        :return: Photometry or error channel
        :rtype: SciPy array
        """
        try:
            if error is False:
                return self.channels[item]
            else:
                return self.errors[item]
        except TypeError:
            if error is False:
                return self.channels[self.ids.index(item)]
            else:
                return self.errors[self.ids.index(item)]

    def get_error(self, item):
        """ Retuns error channel for target given in item.
        :param item: ID of the target. Can be number or label name.
        :return: error associated to photometry of said target
        :rtype: SciPy array
        """
        return self.__getitem__(item, error=True)

    def group1(self):
        """ Returns first grouping of photometry channels.
        :return: list
        """
        return [self.channels[i] for i in range(len(self.channels) - 2) if self.group[i]]

    def group2(self):
        """ Returns second grouping of photometry channels.
        :return: list
        """
        return [self.channels[i] for i in range(len(self.channels) - 2) if not self.group[i]]

    def set_group(self, new_group):
        """ Sets a grouping inside the TimeSeries object. Targets can be grouped or ungrouped together
            as a way to easily perform operations between many of them.
        :param new_group: Mask of 0s and 1s. 1s are the first group. Ex: [0 1 0 0 1 0] defines the first
                            group to contain targets 2 and 5; second group contains targets 1, 3, 4 and 6.
        """
        self.group = new_group

    def errors_group1(self):
        """ Returns error channels of the first group.
        :return: list
        """
        return [self.errors[i] for i in range(len(self.errors) - 2) if self.group[i]]

    def errors_group2(self):
        """ Returns error channels of the second group.
        :return: list
        """
        return [self.errors[i] for i in range(len(self.errors) - 2) if not self.group[i]]

    def set_labels(self, ids):
        """ Sets name labels for each channel
        :param ids: List of label names
        :return: List of labels
        """
        self.ids = ids
        for i in range(len(self.ids)):
            self.labels[self.ids[i]] = self.channels[i]
        return self.labels

    def set_epoch(self, e):
        """ Sets epoch for time serie
        :param e: epoch
        """
        self.epoch = e

    def mean(self, group_id):
        """ Calculates the mean of the channels stored in group group_id. Also calculated for error
            channels. Result is stored in channel timeserie[-group_id]
        :param group_id: int. 1 or 2.
        """
        if group_id > 2:
            warnings.warn("group_id must be 1 or 2 only. Group 2 will be used as default.")
            group = self.group2()
            g_errors = self.errors_group2()
        elif group_id == 1:
            group = self.group1()
            g_errors = self.errors_group1()
        else:
            group = self.group2()
            g_errors = self.errors_group2()

        self.channels[-group_id] = sp.mean(group, axis=0)
        err = sp.zeros((1, len(g_errors[0])))
        for i in range(len(g_errors)):
            err += sp.divide(g_errors[i], group[i])**2
        self.errors[-group_id] = sp.sqrt(err)

        return self.channels[-group_id]

    def median(self, group_id):
        """ Calculates the median of the channels stored in group group_id. Also calculated for error
            channels. Result is stored in channel timeserie[-group_id]
        :param group_id: int. 1 or 2.
        """
        if group_id > 2:
            warnings.warn("group_id must be 1 or 2 only. Group 2 will be used as default.")
            group = self.group2()
            g_errors = self.errors_group2()
        elif group_id == 1:
            group = self.group1()
            g_errors = self.errors_group1()
        else:
            group = self.group2()
            g_errors = self.errors_group2()

        self.channels[-group_id] = sp.median(group, axis=0)
        print(g_errors)
        err = sp.zeros((1, len(g_errors[0])))
        for i in range(len(g_errors)):
            err += sp.divide(g_errors[i], group[i])**2
        self.errors[-group_id] = sp.sqrt(err)

        return self.channels[-group_id]

    def plot(self, label=None, axes=None):
        """Display the timeseries data: flux (with errors) as function of mjd

        :param label: Specify a single star to plot
        :rtype label: basestring

        :rtype: None (and plot display)
        """
        import dataproc as dp
        from datetime import datetime
        from matplotlib import dates

        date_epoch = [datetime.strptime(e, "%Y-%m-%dT%H:%M:%S.%f") for e in self.epoch]
        newepoch = [dates.date2num(dts) for dts in date_epoch]
        #newepoch = self.epoch

        fig, ax, epoch = dp.axesfig_xdate(axes, newepoch)

        if label is None:
            disp = self.labels.keys()
        else:
            disp = [label]

        # TODO check yerr
        for lab in disp:
            if self.__getitem__(lab, error=True) is None:
                yerr = None
            else:
                yerr = self.__getitem__(lab, error=True)

            ax.errorbar(epoch,
                        self.labels[lab],
                        yerr=yerr,
                        marker="o",
                        label=lab)

        ax.set_title("Timeseries Data")
        ax.set_xlabel("MJD")
        ax.set_ylabel("Flux")

        ax.legend()
        import matplotlib.pyplot as plt
        plt.show()
        #return
