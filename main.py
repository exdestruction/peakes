#!/usr/bin/env python3

import argparse
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np  

def main(args):
    filename = args.filename
    method = args.method
    height = args.height
    threshold = args.threshold
    distance = args.distance
    # get data from file provided through command line 
    with open(filename, 'r') as f:
        data = pd.read_csv(f, sep='\t', header=None, skiprows=2)
        x = data.iloc[:,0]
        y = data.iloc[:,1]
        y_log = np.log10(y) 

    if method == 'slow_2t-t':
        peaks = scipy.signal.find_peaks(y_log, height=height, threshold=threshold, distance=distance)
        idxs = peaks[0]
        peaks_x = data.iloc[idxs, 0]
        peaks_y_log = np.log10(data.iloc[idxs, 1])

    fig, ax = plt.subplots()
    ax.plot(x, y_log)
    ax.scatter(peaks_x, peaks_y_log, color='r')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Function for finding peaks of XDR function')
    parser.add_argument('filename')
    parser.add_argument('method')
    parser.add_argument("-he", '--height', default=2.5)
    parser.add_argument("-t", '--threshold', default=.0001)
    parser.add_argument("-d", '--distance', default=100)
    args = parser.parse_args()
    main(args)
