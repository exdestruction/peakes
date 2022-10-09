#!/usr/bin/env python3

import argparse


def main(filename):
    print(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Function for finding peaks of XDR function')
    parser.add_argument('filename')
    args = parser.parse_args()
    main(args.filename)
