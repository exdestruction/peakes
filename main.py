#!/usr/bin/env python3

import argparse
import pandas as pd



def main(filename):
    with open(filename, 'r') as f:
        data = pd.read_csv(f, sep='\t', header=None, skiprows=2)
    print(data)
    print(data.iloc[0])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Function for finding peaks of XDR function')
    parser.add_argument('filename')
    args = parser.parse_args()
    main(args.filename)
