#!/usr/bin/env python3
import os
from enum import Enum
from typing import NamedTuple
import datetime

import pandas as pd
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np  

from utils import als

# 1.54059 is a wavelenght of X-ray in angstrems
CRYSTAL_CONST = 1.54059 
WIDTH = 200
PROMINENCE = 0.2
NUM_PEAKS_SLOW = 30

methods = ['slow', '2t', 'rock']


class Data(NamedTuple):
    name: str
    path: str
    method: str
    peaks: int

class XDR:
    def __init__(self):
        self.prominence = PROMINENCE 
        self.width_region = WIDTH
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.data = self.load_data('data')

    def load_data(self, path):
        def parse_method(name):
            for method in methods:
                if method in name:
                    return method
        def get_peaks(method):
            if method == 'slow':
                peaks = 2
            if method == '2t':
                peaks = 3
            if method == 'rock':
                peaks = 1
            return peaks

        data = []
        for entry in os.scandir(path):
            if entry.is_dir():
                data += self.load_data(entry.path) 
            elif entry.is_file():
                method = parse_method(entry.name)
                peaks = get_peaks(method)
                data.append(Data(name=entry.name, path=entry.path, method=method, peaks=peaks))
        return data

    def report(self):
        for data in self.data:
            self.number_of_peaks = data.peaks
            fig = self.compute(data)
            fig.suptitle(f"{data.name}")
            save_path = f"results/{self.timestamp}/{data.name[:-4]}.png"
            print(f'SAVING: {save_path}')
            try:
                fig.savefig(save_path)
            except FileNotFoundError:
                try:
                    os.mkdir('results')
                except FileExistsError:
                    pass
                os.mkdir(f'results/{self.timestamp}')
                fig.savefig(save_path)

    def compute(self, data):
        self.x, self.y, self.data = self.prepare_data(data.path)
        self.y_log = np.log10(self.y)

        self.y_corrected = self.apply_correction(self.y_log, data.method)
        peaks_idxs = self.find_peaks_idxs(self.y_corrected, self.prominence)
        self.peaks_idxs = self.find_peaks_around(
                self.y_log, 
                peaks_idxs, 
                width_region = self.width_region,
                number_of_peaks=self.number_of_peaks
                )

        self.peaks_x = self.x.loc[self.peaks_idxs]
        self.peaks_y_log = self.y_log.loc[self.peaks_idxs]

        # alpha = 2 tetta in rad
        self.alphas = self.peaks_x.to_numpy() / 180 * np.pi

        if data.method == "slow":
            fig, axs = plt.subplots(1, 2, figsize=(10,5), layout='tight')
            self.plot_data(axs[0])

            # find 20 peaks and thickness
            peaks = self.find_peaks_around(self.y_log, peaks_idxs, width_region=self.width_region*2, number_of_peaks=NUM_PEAKS_SLOW) 
            x = self.x.loc[peaks]
            y_log = self.y_log.loc[peaks]
            thickness = self.get_thickness(x) 
            axs[0].scatter(x, y_log, color='blue')
            axs[0].text(.05, .95, f'thickness={thickness:.5f} nm',
                horizontalalignment='left',
                verticalalignment='top',
                transform=axs[0].transAxes)

            self.plot_closer_area(axs[1])
            return fig
            
        elif data.method == '2t':
            fig, axs = plt.subplot_mosaic([["first", "first", "first"],
                                            ["left", "center", "right"]], figsize=(15,10))
            self.plot_data(axs['first'])
            self.draw_regression([axs['left'], axs['center'], axs['right']])
            return fig

        elif data.method == "rock":
            fig, axs = plt.subplots(1, 1, figsize=(5,5))
            self.plot_data(axs) 
            self.plot_rock(axs)
            return fig


    def plot_rock(self, ax):
        peak_x = self.x.loc[self.peaks_idxs].to_numpy()[0]
        peak_y = self.y.loc[self.peaks_idxs].to_numpy()[0]
        peak_yhalf = peak_y / 2
        peak_yhalflog = np.log10(peak_yhalf)
        data = self.data.to_numpy()
        # find 2 nearest points to yhalf
        _, index = scipy.spatial.KDTree(data).query([peak_x, peak_yhalf], k=2)
        points = data[index]
        x_coords = points[:,0]
        difference = abs(x_coords[:-1] - x_coords[1:])
    
        # plotting
        ax.text(.10, .85, f'FWHM={float(difference):<5f}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)
        ax.text(.10, .95, f'HM={peak_yhalflog:.5f}',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)

    def plot_data(self, ax):
        ax.plot(self.x, self.y_log)
        ax.plot(self.x, self.y_corrected)
        ax.scatter(self.peaks_x, self.peaks_y_log, color = 'r')
    
    def draw_regression(self, axs):
        half_alphas = np.reshape(self.alphas, (-1, self.number_of_peaks)) / 2
        NR = 0.5 * np.power(np.cos(half_alphas), 2) / np.sin(half_alphas) + (np.power(np.cos(half_alphas), 2) / half_alphas)
        proj_num = np.array(range(1, NR.shape[0]+1), ndmin=2).transpose()
        I = CRYSTAL_CONST / (2 * np.sin(half_alphas)) * proj_num
        num_areas = NR.shape[1]
        for i in range(num_areas):
            # this is to deal with plots, if only one metric/plot is present
            try:
                ax = axs[i]
            except TypeError:
                ax = axs
            x = NR[:, i]
            y = I[:, i]
            a, b, r, p, std_err = scipy.stats.linregress(x, y)
            ax.scatter(x,y , color='red')
            ax.plot(x, a*x+b, label=f'y = {a:.5f} * x + {b:.5f}\nr^2={r**2:.5f}')
            ax.set_title(f'Peak #{i}')
            ax.set_xlabel('NR')
            ax.set_ylabel('I')
            ax.legend()
            ax.grid()

    @staticmethod
    def get_thickness(peaks_x):
        alphas = peaks_x.to_numpy() / 180 * np.pi
        thickness = np.mean(CRYSTAL_CONST / 2 / (np.sin(alphas[1:]) - np.sin(alphas[:-1]))) / 10
        return thickness


    def plot_closer_area(self, ax):
        x = self.x.loc[self.peaks_idxs[0]:self.peaks_idxs[-1]]
        y_log = self.y_log.loc[self.peaks_idxs[0]:self.peaks_idxs[-1]]
        peaks = scipy.signal.find_peaks(y_log)

        peaks_x = x.iloc[peaks[0]]
        peaks_y = y_log.iloc[peaks[0]]

        thickness = XDR.get_thickness(peaks_x)

        # plotting
        ax.plot(x, y_log)
        ax.scatter(peaks_x, peaks_y, color='r')
        ax.text(.05, .95, f'thickness={thickness:.5f} nm',
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)


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

        return x, y, data

    @staticmethod 
    def apply_correction(y_log, method):
        # applying baseline correction "asymmetric least square method"
        if method == 'rock':
            y_res = als(y_log.to_numpy(), 1e6)
        else:
            y_res = als(y_log.to_numpy())

        return y_res

    @staticmethod
    def find_peaks_idxs(y, prominence=0):
        peaks = scipy.signal.find_peaks(y, prominence=prominence)
        peaks_idxs = peaks[0]
        return peaks_idxs
    
    @staticmethod
    def find_peaks_around(y_log, peaks, width_region=150, number_of_peaks=None):
        indexes_peaks_overall = []
        for idx in peaks:
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
    # args = parse_cli_args()
    xdr = XDR()
    xdr.report()


