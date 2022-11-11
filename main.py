#!/usr/bin/env python3

import pandas as pd
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np  

from utils import als, draw_regression, parse_cli_args

# 1.54059 is a wavelenght of X-ray in angstrems
CRYSTAL_CONST = 1.54059 

class XDR:
    def __init__(self, args):
        self.filename = args.filename
        self.method = args.method
        self.prominence = args.prominence
        self.number_of_peaks = args.number_of_peaks
        self.width_region = args.width 

    def compute_all(self):
        # files = self.load_data()
        x, y = self.prepare_data(self.filename)
        y_log = np.log10(y)
        y_corrected = self.apply_correction(y_log, self.method)
        peaks_idxs_corrected = self.find_peaks_idxs(y_corrected, self.prominence)
        peaks_idxs = self.find_peaks_around_corrected(
                y_log, 
                peaks_idxs_corrected, 
                width_region = self.width_region,
                number_of_peaks=self.number_of_peaks)

        peaks_x = x.loc[peaks_idxs]
        peaks_y_log = y_log.loc[peaks_idxs]

        # alpha = 2 tetta in rad
        alphas = peaks_x.to_numpy() / 180 * np.pi


        fig, ax = plt.subplots()
        ax.plot(x, y_log)
        ax.plot(x, y_corrected)
        ax.scatter(peaks_x, peaks_y_log, color='r')
        ax.set_xlabel('2 tetta', fontsize=12) 
        ax.set_ylabel('Intensity', fontsize=12)

        if self.method == 'slow_2t-t':
            thickness = np.mean(CRYSTAL_CONST / 2 / (np.sin(alphas[1:]) - np.sin(alphas[:-1]))) / 10
            # plotting
            ax.text(.05, .95, f'thickness={thickness:.5f} nm',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)

        elif self.method == "2t-t":
            half_alphas = np.reshape(alphas, (-1, number_of_peaks)) / 2
            NR = 0.5 * np.power(np.cos(half_alphas), 2) / np.sin(half_alphas) + (np.power(np.cos(half_alphas), 2) / half_alphas)
            proj_num = np.array(range(1, NR.shape[0]+1), ndmin=2).transpose()
            I = CRYSTAL_CONST / (2 * np.sin(half_alphas)) * proj_num   
            # plotting
            regress_fig = draw_regression(NR, I)

        elif self.method == "rock":
            peak_y = y.loc[peaks_idxs].to_numpy()[0]
            peak_yhalf = peak_y / 2
            peak_yhalflog = np.log10(peak_yhalf)
            # plotting
            ax.text(.05, .95, f'FWHM={peak_yhalflog:.5f}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)
        plt.show()
        # raise

    def compute_slow(self):
        pass

    def load_data(self):
        # get all the data from ./data
        pass

    @staticmethod
    def prepare_data(filename):
        # get data from file provided through command line 
        with open(filename, 'r') as f:
            data = pd.read_csv(f, sep='\t', header=None, skiprows=2)

        # drop rows, where any values == 0 
        data = data[~(data == 0).any(axis=1)]
        # x is 2 tetta
        x = data.iloc[:,0]
        y = data.iloc[:,1]

        return x, y

    @staticmethod 
    def apply_correction(y_log, method):
        # applying baseline correction "asymmetric least square method"
        if method == 'rock':
            y_res = als(y_log.to_numpy(), 1e6)
        else:
            y_res = als(y_log.to_numpy())

        return y_res

    @staticmethod
    def find_peaks_idxs(y, prominence):
        peaks = scipy.signal.find_peaks(y, prominence=prominence)
        peaks_idxs = peaks[0]
        return peaks_idxs
    
    @staticmethod
    def find_peaks_around_corrected(y_log, peaks_corrected, width_region=150, number_of_peaks=3):
        indexes_peaks_overall = []
        for idx in peaks_corrected:
            area_around = y_log.iloc[idx - width_region:idx + width_region]
            area_peaks = scipy.signal.find_peaks(area_around)
            peaks_data = area_around.iloc[area_peaks[0]]
            prominence = scipy.signal.peak_prominences(area_around, area_peaks[0])

            # pandas.Series to pandas.DataFrame to be able to acces indexes
            peaks_data = peaks_data.to_frame()
            # add column with prominence
            peaks_data['prominence'] = prominence[0]

            # number_of_peaks of most significant peaks
            peaks_data_sorted = peaks_data.sort_values('prominence')
            peaks_single_area = peaks_data_sorted.tail(number_of_peaks)

            idxs_peaks_single_area = np.sort(peaks_single_area.index.to_numpy())
            single_area_peaks = idxs_peaks_single_area.tolist()
            indexes_peaks_overall += single_area_peaks
        
        return indexes_peaks_overall

if __name__ == '__main__':
    args = parse_cli_args()
    xdr = XDR(args)
    xdr.compute_all()


