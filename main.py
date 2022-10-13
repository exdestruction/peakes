#!/usr/bin/env python3

import argparse
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np  

CRYSTAL_CONST = 1.54059 / 2

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

def main(args):
    filename = args.filename
    method = args.method
    height = args.height
    threshold = args.threshold
    distance = args.distance
    prominence = args.prominence

    # get data from file provided through command line 
    with open(filename, 'r') as f:
        data = pd.read_csv(f, sep='\t', header=None, skiprows=2)

    # drop rows, where any values == 0 
    data = data[~(data == 0).any(axis=1)]
    x = data.iloc[:,0]
    y = data.iloc[:,1]
    y_log = np.log10(y)

    # applying baseline correction "asymmetric least square method"
    y_res = als(y_log.to_numpy()) 


    peaks = scipy.signal.find_peaks(y_res, prominence=prominence)
    idxs = peaks[0]
    peaks_x = data.iloc[idxs, 0]
    peaks_y_log = y_res[idxs]
    alphas = peaks_x /180 * np.pi

    if method == 'slow_2t-t':
        thickness = np.mean(CRYSTAL_CONST / (np.sin(alphas[1:]) - np.sin(alphas[:-1]))) / 10
    elif method == "2t-t":
        thickness = CRYSTAL_CONST

    fig, ax = plt.subplots()
    ax.plot(x, y_log)
    ax.plot(x, y_res)
    ax.scatter(peaks_x, peaks_y_log, color='r')
    ax.text(.05, .95, f'thickness={thickness:.5f} nm',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)
    ax.set_xlabel('2 tetta / deg', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Function for finding peaks of XDR function')
    parser.add_argument('filename')
    parser.add_argument('method')
    parser.add_argument("-he", '--height', default=2.5)
    parser.add_argument("-t", '--threshold', default=.1)
    parser.add_argument("-d", '--distance', default=100)
    parser.add_argument("-p", "--prominence",type=int, default=0.5)
    args = parser.parse_args()
    main(args)


