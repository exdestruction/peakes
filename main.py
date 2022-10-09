#!/usr/bin/env python3

import argparse
import pandas as pd
import scipy



def main(args):
    filename = args.filename
    method = args.method
    height = args.height
    threshold = args.threshold
    distance = args.distance
    with open(filename, 'r') as f:
        data = pd.read_csv(f, sep='\t', header=None, skiprows=2)
    if method == 'slow_2t-t':
        peaks = scipy.signal.find_peaks(data.iloc[:,1], height=height, threshold=threshold, distance=distance)
       


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Function for finding peaks of XDR function')
    parser.add_argument('filename')
    parser.add_argument('method')
    parser.add_argument("-he", '--height', default=2.5)
    parser.add_argument("-t", '--threshold', default=.0001)
    parser.add_argument("-d", '--distance', default=100)
    args = parser.parse_args()
    main(args)
