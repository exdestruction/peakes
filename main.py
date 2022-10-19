#!/usr/bin/env python3

import argparse
import itertools

import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np  

# 1.54059 is a wavelenght of X-ray in angstrems
CRYSTAL_CONST = 1.54059 

def als(y, lam=1e9, p=0.5, itermax=10):
    r"""
    Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm (P. Eilers, H. Boelens 2005)

    Baseline Correction with Asymmetric Least Squares Smoothing
    based on https://github.com/vicngtor/BaySpecPlots

    Baseline Correction with Asymmetric Least Squares Smoothing
    Paul H. C. Eilers and Hans F.M. Boelens
    October 21, 2005

    Description from the original documentation:

    Most baseline problems in instrumental methods are characterized by a smooth
    baseline and a superimposed signal that carries the analytical information: a series
    of peaks that are either all positive or all negative. We combine a smoother
    with asymmetric weighting of deviations from the (smooth) trend get an effective
    baseline estimator. It is easy to use, fast and keeps the analytical peak signal intact.
    No prior information about peak shapes or baseline (polynomial) is needed
    by the method. The performance is illustrated by simulation and applications to
    real data.


    Inputs:
        y:
            input data 
        lam:
            The larger lambda is, the smoother the resulting background - z
        p:
            wheighting deviations. 0.5 = symmetric, <0.5: negative
            deviations are stronger suppressed
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector

    """
    L = len(y)
    D = scipy.sparse.eye(L, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]
    D = D.T
    w = np.ones(L)
    for i in range(itermax):
        W = scipy.sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.T)
        z = scipy.sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def draw_regression(NR, I):
    num_areas = NR.shape[1]
    fig, axs = plt.subplots(num_areas, constrained_layout=True)
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
    return fig

def main(args):
    filename = args.filename
    method = args.method
    prominence = args.prominence
    number_of_peaks = args.number_of_peaks
    width_region = args.width 

    # get data from file provided through command line 
    with open(filename, 'r') as f:
        data = pd.read_csv(f, sep='\t', header=None, skiprows=2)

    # drop rows, where any values == 0 
    data = data[~(data == 0).any(axis=1)]
    # x is 2 tetta
    x = data.iloc[:,0]
    y = data.iloc[:,1]
    y_log = np.log10(y)

    # applying baseline correction "asymmetric least square method"
    if method == 'rock':
        y_res = als(y_log.to_numpy(), 1e6)
    else:
        y_res = als(y_log.to_numpy()) 

    # peaks of preprocessed data
    peaks_baselined = scipy.signal.find_peaks(y_res, prominence=prominence)
    peaks_baselined_idxs = peaks_baselined[0]

    indexes_peaks_overall = []
    for idx in peaks_baselined_idxs:
        area_around = y_log.iloc[idx - width_region:idx + width_region]
        area_peaks = scipy.signal.find_peaks(area_around)
        area_peaks_idx = area_peaks[0]
        peaks_data = area_around.iloc[area_peaks_idx]
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

    peaks_x = data.loc[indexes_peaks_overall, 0]
    peaks_y_log = y_log.loc[indexes_peaks_overall]

    # alpha = 2 tetta in rad
    alphas = peaks_x.to_numpy() / 180 * np.pi


    fig, ax = plt.subplots()
    ax.plot(x, y_log)
    ax.plot(x, y_res)
    ax.scatter(peaks_x, peaks_y_log, color='r')
    ax.set_xlabel('2 tetta', fontsize=12) 
    ax.set_ylabel('Intensity', fontsize=12)

    if method == 'slow_2t-t':
        thickness = np.mean(CRYSTAL_CONST / 2 / (np.sin(alphas[1:]) - np.sin(alphas[:-1]))) / 10
        # plotting
        ax.text(.05, .95, f'thickness={thickness:.5f} nm',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)

    elif method == "2t-t":
        half_alphas = np.reshape(alphas, (-1, number_of_peaks)) / 2
        NR = 0.5 * np.power(np.cos(half_alphas), 2) / np.sin(half_alphas) + (np.power(np.cos(half_alphas), 2) / half_alphas)
        proj_num = np.array(range(1, NR.shape[0]+1), ndmin=2).transpose()
        I = CRYSTAL_CONST / (2 * np.sin(half_alphas)) * proj_num   
        # plotting
        regress_fig = draw_regression(NR, I)

    elif method == "rock":
        peak_y = data.loc[indexes_peaks_overall, 1].to_numpy()[0]
        peak_yhalf = peak_y / 2
        peak_yhalflog = np.log10(peak_yhalf)
        # plotting
        ax.text(.05, .95, f'FWHM={peak_yhalflog:.5f}',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Function for finding peaks of XDR function')
    parser.add_argument('filename')
    parser.add_argument('method', help="One of 3 methods: slow_2t-t, 2t-t, rock")
    parser.add_argument("number_of_peaks", type=int, help='Number of peaks to be found on one area of interest')
    parser.add_argument("-p", "--prominence",type=float, default=0.3, help="Defines how distinguishable important peaks should be. Default=0.3")
    parser.add_argument("-w", "--width", type=int, default=200, help='Defines width of area of interest. Default=200 (points)')
    args = parser.parse_args()
    main(args)


