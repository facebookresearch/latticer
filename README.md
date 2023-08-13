The accompanying software can reproduce all figures in the associated paper,
"An efficient algorithm for integer lattice reduction." This repository also
provides LaTeX and BibTeX sources for replicating the paper.

The main files in the repository are the following:

``tex/paper.pdf``
PDF version of the paper

``tex/paper.tex``
LaTeX source for the paper

``tex/supplement.pdf``
PDF version of the supplementary materials

``tex/supplement.tex``
LaTeX source for the supplementary materials

``tex/shared.tex``
LaTeX source shared by the paper and supplementary materials

``tex/lattice.bib``
BibTeX source for the paper and supplementary materials

``tex/siamplain.bst``
BibTeX style file

``tex/siamart220329.cls``
LaTeX document class file

``codes/graph.sh``
Shell script that reproduces all figures for the paper when run in ``codes/``.

``codes/graph.c``
ISO C code that graph.sh compiles and runs to generate data for the figures.

``codes/graph.py``
Python script graph.sh runs to reproduce the figs. using the output of graph.c.

Both graph.sh and graph.c are configured for running on the MacOS. The latter
depends on BLAS, which is available under the MacOS in Xcode as "Accelerate"
(a framework for optimized computations). The compiler used is the alias "gcc"
(under Xcode "gcc" points to the Clang LLVM), using optimization level "-O3"
by default. Another operating system might require modification to these lines
in graph.sh, modifying ``gcc graph.c -o graph.exe -O3 -framework Accelerate``
to ``gcc graph.c -o graph.exe -O3 -lblas -lm`` (or something similar).
Similarly, graph.c includes a preprocessing directive in the early line,
``#include <Accelerate/Accelerate.h>``, which includes the header for BLAS ...
whereas most operating systems would need the preprocessing directive,
``#include <cblas.h>``, instead. Both graph.sh and graph.c try to make these
modifications automatically, but different systems might require different
modifications.

********************************************************************************

License

This latticer software is licensed under the LICENSE file (the MIT license) in
the root directory of this source tree.
