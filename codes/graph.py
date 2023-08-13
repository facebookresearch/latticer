#!/usr/bin/env python3
"""
Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

This script reproduces all figures for a specified value of the parameter delta
from the Lovasz criterion in the LLL algorithm for the paper, "An efficient
algorithm for integer lattice reduction." The script assumes that graph.c
has been compiled and run twice, once with the argument setting large = 0
and once with the argument setting large = 1; if the compiled graph.c gets run
only once, then this script will generate only the plots corresponding to large
for the single run. The other parameter delta for running the compiled graph.c
specifies delta, of course.
"""


import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def readfile(filename):
    """
    Reads the file of floats specified by filename

    Reads line-by-line the file specified by filename under the assumption that
    each line consists of ASCII text giving a single floating-point number.

    Parameters
    ----------
    filename : string
        name of the file to read

    Returns
    -------
    list of floats
    """
    a = []
    with open(filename, 'r') as f:
        for line in f:
            a.append(float(line))
    return a


# Read the files with the abscissae.
for filename in os.listdir():
    if filename[-5:] == 'x.txt':
        if '00' in filename:
            x00 = readfile(filename)
        elif '01' in filename:
            x01 = readfile(filename)
        elif '10' in filename:
            x10 = readfile(filename)
        elif '11' in filename:
            x11 = readfile(filename)

# Process all files in the working directory.
for filename in os.listdir():
    if (filename[-4:] == '.txt' and filename[-5] != 'x'
            and 'stddev' not in filename and 'max' not in filename
            and 'frobmin' not in filename and 'minmin' not in filename):
        # Read the file with the ordinates.
        y = readfile(filename)
        # Read the file with the standard deviations and extrema, if relevant.
        if 'mean' in filename:
            z = readfile(filename.replace('mean', 'stddev'))
            ymin = readfile(filename.replace('mean', 'min'))
            ymax = readfile(filename.replace('mean', 'max'))
        # Select the appropriate abscissae.
        if '00' in filename:
            x = x00
        elif '01' in filename:
            x = x01
        elif '10' in filename:
            x = x10
        elif '11' in filename:
            x = x11
        # Construct the title for the plot.
        if 'frob' in filename:
            title = 'reduction in Frob. norm'
        elif 'min' in filename:
            title = 'reduction in min. norm'
        else:
            title = 'running-time'
        if filename[-5] == '1':
            title += ' by LLL'
        if filename[-5] == '2':
            title += ' by ours'
        if 'lll' in filename:
            title += ' by LLL'
        if 'iterate' in filename:
            title += ' by ours'
        if 'mean' in filename or 'lll' in filename:
            title += ' run'
            if 'lll' in filename or '2' not in filename:
                title += ' once'
            if 'mean' in filename and '2' in filename:
                title += ' after LLL'
        if 'iterate' in filename:
            title += ' run after LLL run once'
        if 'multi' in filename and '2' not in filename:
            title += ' run repeatedly with ours'
        if 'multi' in filename and '2' in filename:
            title += ' after reps. of ours+LLL'
        # Generate the plot.
        plt.figure(figsize=(4.3333333, 4.333333))
        if 'mean' in filename:
            w = [np.array(y) - np.array(ymin), np.array(ymax) - np.array(y)]
            plt.errorbar(x, y, w, fmt='k', capsize=10)
        else:
            plt.plot(x, y, 'k')
        # Label the axes.
        plt.ticklabel_format(style='plain')
        if 't_' in filename:
            plt.ylabel('time in seconds')
            if not (('01' in filename and 'lll' in filename)
                    or ('11' in filename and 'lll' in filename)
                    or ('01' in filename and 'iterate' in filename)
                    or ('11' in filename and 'iterate' in filename)):
                plt.yscale('log')
                plt.ylim((1e-6, 1e2))
        else:
            plt.ylabel('fraction remaining after reduction')
        if '00' in filename or '10' in filename:
            plt.xlabel('$n$')
            plt.xscale('log')
        if '01' in filename or '11' in filename:
            plt.xlabel('$p$')
        plt.xticks(x, [str(int(k)) for k in x])
        plt.minorticks_off()
        # Title the plot.
        plt.title(title, fontsize=10)
        # Clean up the whitespace in the plot.
        plt.tight_layout()
        # Save the plot.
        plt.savefig(filename[:-4] + '.pdf', bbox_inches='tight')
        plt.close()
