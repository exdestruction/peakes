import argparse
import scipy.sparse
import matplotlib.pyplot as plt
import numpy as np

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


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Function for finding peaks of XDR function')
    parser.add_argument('filename')
    parser.add_argument('method', help="One of 3 methods: slow_2t-t, 2t-t, rock")
    parser.add_argument("number_of_peaks", type=int, help='Number of peaks to be found on one area of interest')
    parser.add_argument("-p", "--prominence",type=float, default=0.3, help="Defines how distinguishable important peaks should be. Default=0.3")
    parser.add_argument("-w", "--width", type=int, default=200, help='Defines width of area of interest. Default=200 (points)')
    args = parser.parse_args()
    return args
