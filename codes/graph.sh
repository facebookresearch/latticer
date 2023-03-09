#!/bin/sh
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file
# in the root directory of this source tree.
#
# This script reproduces all figures in the paper, "An efficient algorithm
# for integer lattice reduction." The script compiles graph.c (which requires
# BLAS -- replace "-framework Accelerate" with the appropriate linker options
# if running on an operating system other than MacOS) using the gcc compiler
# with the highest level of optimization, -O3, enabled. The script then deletes
# all txt, pdf, and out files in the working directory. Finally, the script
# runs the compiled executable, twice with the parameter delta from the Lovasz
# criterion in the LLL algorithm set to 1 - 1e-15, and twice with the parameter
# delta set to 1 - 1e-1. The script creates anew subdirectories named "1-1e-15"
# and "1-1e-1", moving the associated txt and pdf files into the corresponding
# subdirectories. The first runs set the parameter large for graph.c to 0,
# while the second runs set large to 1.
#

# Compile graph.c.
rm -f graph.exe
case "$(uname -s)" in
    Darwin*)
        gcc graph.c -o graph.exe -O3 -framework Accelerate
        ;;
    Linux*)
        gcc graph.c -o graph.exe -O3 -lblas -lm -Wno-format-truncation
        ;;
esac

# Delete all txt, pdf, and out files in the working directory.
rm -f ./*.txt
rm -f ./*.pdf
rm -f ./*.out

# Run with delta = 1 - 1e-15 (the default setting).
rm -rf 1-1e-15
mkdir 1-1e-15
./graph.exe 0 | tee 1-1e-15_0.out
./graph.exe 1 | tee 1-1e-15_1.out
python graph.py
mv ./*.txt 1-1e-15/
mv ./*.pdf 1-1e-15/

# Run with delta = 1 - 1e-1.
rm -rf 1-1e-1
mkdir 1-1e-1
./graph.exe 0 0.9 | tee 1-1e-1_0.out
./graph.exe 1 0.9 | tee 1-1e-1_1.out
python graph.py
mv ./*.txt 1-1e-1/
mv ./*.pdf 1-1e-1/
