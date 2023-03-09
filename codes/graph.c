/*
Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

N.B.: This code leverages and depends on BLAS. The preprocessor directive below
assumes that the operating system is MacOS. For other operating systems,
try replacing the line, "#include <Accelerate/Accelerate.h>", with the line,
"#include <cblas.h>".

This ISO C code reproduces all figures in the paper, "An efficient algorithm
for integer lattice reduction." The associated compiled executable will accept
one or two command-line options. The first is required and specifies whether
the entries of the basis vectors being processed start with relatively large
values (large = 1) or with relatively smaller values (large = 0). "Large" means
that the entries' range is the Mersenne prime q = 2^{31} - 1, while the smaller
case uses the Mersenne prime q = 2^{13} - 1. The second command-line option --
which is optional -- specifies the value of the parameter delta in the Lovasz
criterion of the LLL algorithm (3e-16 < delta < 1); delta defaults to 1-1e-15.

The main program reduces basis vectors that form the rows of a matrix
(q*Id, 0; R, Id), where Id is the identity matrix and R is a random matrix
whose entries are drawn independently and identically distributed uniformly
over the integers -(q-1)/2, -(q-3)/2, ..., (q-3)/2, (q-1)/2. The ordering of
the rows gets randomized via a uniformly random permutation prior to reduction.
The dimensions of R are lesser x larger, with the dimension of the full matrix
(q*Id, 0; R, Id) being n = lesser + larger; larger = 2 * lesser, with lesser
= 2, 4, 8, ..., 2^{NUMEX}, where NUMEX is the constant defined in main().

The main program runs two series of experiments, the first varying n, and
the second varying the parameter p that defines the objective function which
the algorithm of the new paper optimizes (namely, the sum of the p-th powers
of the Euclidean norms of the basis vectors). In both series, each experiment
runs two different combinations of the LLL algorithm and that from iterate().
The first simply permutes the rows at random, then runs LLL, and finishes
by applying iterate() to the results of the LLL. This first series runs
INSTANCES differently seeded random instantiations, averaging the results and
calculating their minima, maxima, and standard deviations. The second series
runs INSTANCES sequences of randomly permuting the rows followed by LLL
followed by iterate(), all in series (with the outputs of the previous stage
becoming the inputs of the subsequent stage). In both series, INSTANCES is
the constant defined in main(). The main program writes out to ASCII text files
all outputs appropriate for plotting with graph.py using a rather idiosyncratic
scheme for naming the files. The main program returns 0 upon success; a return
of -2 indicates a problem parsing the command-line options, while -3 indicates
that the number of command-line options was wrong (there should be either 1
or 2 command-line options specified). main() returns -1 when the LLL algorithm
fails to produce a reduced basis meeting the specified LLL criteria.

The codes include the following functions:

frobnorm
    Calculate the Frobenius norm (sqrt of the sum of squares of entries).
gs
    Run Gram-Schmidt without normalization.
iterate
    Reduce Euclidean norms via projections until they stop changing.
lll
    Conduct iterations of the Lenstra-Lenstra-Lovasz (LLL) algorithm.
lll_check
    Check that the Lenstra-Lenstra-Lovasz (LLL) criteria are satisfied.
llliteratest
    Test lll followed by iterate.
maximum
    Calculate the maximum of the entries of the input vector.
mean
    Calculate the mean of the entries of the input vector.
minimum
    Calculate the minimum of the entries of the input vector.
minnorm
    Calculate the Euclidean norm of the shortest row.
randombasis
    Form a basis for a lattice at random, of the form (q*Id, 0; R, Id).
randrows
    Order the rows at random via the Fisher-Yates-Durstenfield-Knuth shuffle.
runinstances
    Run llliteratetest several times alone and the same number in series.
stddev
    Calculate the standard deviation of the entries of the input vector.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#if __APPLE__
    #include <TargetConditionals.h>
    #if TARGET_OS_MAC
        #include <Accelerate/Accelerate.h>
    #else
    #   error "This Apple OS is not MacOS."
    #endif
#else
    #include <cblas.h>
#endif

typedef double int53_t;


void randrows(int m, int n, int53_t (*a)[m]) {
    /* Order the rows at random via Fisher-Yates-Durstenfield-Knuth shuffle.

    Parameters
    ----------
    m -- second dimension of a
    n -- first dimension of a
    a -- two-dimensional array whose rows get randomized

    Returns
    -------
    a -- two-dimensional array whose rows got randomized
    */
    int j, k;

    for (j = n - 1; j >= 1; j--) {
        /* Choose an integer uniformly at random from 0 to j. */
        k = 1. * (j + 1) * random() / (RAND_MAX + 1e-4);
        /* Swap rows j and k. */
        cblas_dswap(m, a[j], 1, a[k], 1);
    }
}


void gs(int m, int n, double (*a)[m]) {
    /* Run Gram-Schmidt without normalization.

    Parameters
    ----------
    m -- second dimension of a
    n -- first dimension of a
    a -- two-dimensional array whose rows get orthogonalized

    Returns
    -------
    a -- two-dimensional array whose rows got orthoganalized
    */
    double b, c, d;
    int j, k, l;
    char changed, count;

    for (j = 1; j < n; j++) {
        /* Reorthogonalize several times to combat roundoff errors. */
        count = 0;
        do {
            changed = 0;
            for (k = 0; k < j; k++) {
                /* Calculate the inner product of a[k][:] with itself. */
                b = cblas_ddot(m, a[k], 1, a[k], 1);
                /* Calculate the inner product of a[j][:] with a[k][:]. */
                c = cblas_ddot(m, a[j], 1, a[k], 1);
                /* Subtract off the projection of a[j][:] onto a[k][:]. */
                cblas_daxpy(m, -c / b, a[k], 1, a[j], 1);
                /* Calculate the inner product of a[j][:] with itself. */
                d = cblas_ddot(m, a[j], 1, a[j], 1);
                if (c * c > 1e-28 * b * d) changed = 1;
            }
            if (!changed) count++;
        } while (count < 2);
    }
}


void lll(int m, int n, int53_t (*a)[m], double delta) {
    /* Conduct iterations of the Lenstra-Lenstra-Lovasz (LLL) algorithm.

    Parameters
    ----------
    m -- second dimension of a
    n -- first dimension of a
    a -- two-dimensional array whose rows get reduced
    delta -- parameter in the Lovasz criterion (0 < delta < 1)

    Returns
    -------
    a -- two-dimensional array whose rows got reduced
    */
    double (*b)[m] = malloc(sizeof(double[n][m]));
    int j, k;
    double f, g, h;
    char changed, count;

    /* Copy a into b. */
    for (k = 0; k < n; k++)
        cblas_dcopy(m, a[k], 1, b[k], 1);
    /* Run Gram-Schmidt (without normalization) on b. */
    gs(m, n, b);
    /* Conduct the LLL iterations. */
    j = 1;
    while (j < n) {
        /* Reduce the basis. */
        for (k = j - 1; k >= 0; k--) {
            /* Calculate the inner product of b[k][:] with itself. */
            f = cblas_ddot(m, b[k], 1, b[k], 1);
            /* Calculate the inner product of a[j][:] with b[k][:]. */
            g = cblas_ddot(m, a[j], 1, b[k], 1);
            /* Subtract off the rounded projection of a[j][:] onto a[k][:]. */
            cblas_daxpy(m, -round(g / f), a[k], 1, a[j], 1);
        }
        /* Calculate the inner product of b[j - 1][:] with itself. */
        f = cblas_ddot(m, b[j - 1], 1, b[j - 1], 1);
        /* Calculate the inner product of a[j][:] with b[j - 1][:]. */
        g = cblas_ddot(m, a[j], 1, b[j - 1], 1);
        /* Calculate the inner product of b[j][:] with itself. */
        h = cblas_ddot(m, b[j], 1, b[j], 1);
        /* Calculate the projection coefficient. */
        g /= f;
        /* Swap vectors if necessary, testing the Lovasz criterion first. */
        if (h < (delta - g * g) * f) {
            /* Swap a[j - 1][:] and a[j][:]. */
            cblas_dswap(m, a[j - 1], 1, a[j], 1);
            /* Recompute b[j - 1][:] from scratch for numerical stability. */
            cblas_dcopy(m, a[j - 1], 1, b[j - 1], 1);
            if (j > 1) {
                /* Reorthogonalize several times to combat roundoff errors. */
                count = 0;
                do {
                    changed = 0;
                    for (k = 0; k < j - 1; k++) {
                        /* Compute the inner product of b[k][:] with itself. */
                        f = cblas_ddot(m, b[k], 1, b[k], 1);
                        /* Compute the dot prod. of b[j - 1][:] and b[k][:]. */
                        g = cblas_ddot(m, b[j - 1], 1, b[k], 1);
                        /* Subtract off the proj. of b[j - 1][:] on b[k][:]. */
                        cblas_daxpy(m, -g / f, b[k], 1, b[j - 1], 1);
                        /* Compute the dot prod. of b[j - 1][:] with itself. */
                        h = cblas_ddot(m, b[j - 1], 1, b[j - 1], 1);
                        if (g * g > 1e-28 * f * h) changed = 1;
                    }
                    if (!changed) count++;
                } while (count < 2);
            }
            /* Recompute b[j][:] from scratch for numerical stability. */
            cblas_dcopy(m, a[j], 1, b[j], 1);
            /* Reorthogonalize several times to combat roundoff errors. */
            count = 0;
            do {
                changed = 0;
                for (k = 0; k < j; k++) {
                    /* Calculate the inner product of b[k][:] with itself. */
                    f = cblas_ddot(m, b[k], 1, b[k], 1);
                    /* Calculate the inner product of b[j][:] with b[k][:]. */
                    g = cblas_ddot(m, b[j], 1, b[k], 1);
                    /* Subtract off the projection of b[j][:] onto b[k][:]. */
                    cblas_daxpy(m, -g / f, b[k], 1, b[j], 1);
                    /* Calculate the inner product of b[j][:] with itself. */
                    h = cblas_ddot(m, b[j], 1, b[j], 1);
                    if (g * g > 1e-28 * f * h) changed = 1;
                }
                if (!changed) count++;
            } while (count < 2);
            /* Decrement j. */
            if (j > 1) j--;
        } else {
            j++;
        }
    }

    free(b);
}


char lll_check(int m, int n, int53_t (*a)[m], double delta) {
    /* Check that the Lenstra-Lenstra-Lovasz (LLL) criteria are satisfied.

    Parameters
    ----------
    m -- second dimension of a
    n -- first dimension of a
    a -- two-dimensional array whose rows should satisfy the LLL criteria
    delta -- parameter in the Lovasz criterion (0 < delta < 1)

    Returns
    -------
    1 if the criteria are satisfied, 0 if the criteria are not satisfied
    */
    double (*b)[m] = malloc(sizeof(double[n][m]));
    char correct = 1;
    int j, k;
    double f, g, absmax, fold, ratio;

    /* Copy a into b. */
    for (k = 0; k < n; k++)
        cblas_dcopy(m, a[k], 1, b[k], 1);
    /* Run Gram-Schmidt (without normalization) on b. */
    gs(m, n, b);
    /* Compute the largest proj. of each row of a on each earlier row of b. */
    absmax = 0;
    for (k = 1; k < n; k++) {
        for (j = 0; j < k; j++) {
            /* Calculate the inner product between b[j][:] and itself. */
            f = cblas_ddot(m, b[j], 1, b[j], 1);
            /* Calculate the inner product between a[k][:] and b[j][:]. */
            g = cblas_ddot(m, a[k], 1, b[j], 1);
            if (fabs(g / f) > absmax) absmax = fabs(g / f);
        }
    }
    printf("absmax = %f\n", absmax);
    if (absmax > 0.5 + 1e-1) correct = 0;
    /* Compute the least ratio of the squares of successive basis vectors. */
    ratio = 1;
    for (k = 0; k < n; k++) {
        /* Calculate the inner product between b[k][:] and itself. */
        f = cblas_ddot(m, b[k], 1, b[k], 1);
        /* Update the ratios. */
        if (k > 0) {
            if (f / fold < ratio) ratio = f / fold;
        }
        fold = f;
    }
    printf("ratio = %f\n", ratio);
    /* The ratio arising from the Lovasz condition with parameter delta */
    /* should be at most  delta - 0.25. */
    if (ratio <= delta - 1. / 4 - 1e-1) correct = 0;

    free(b);

    return correct;
}


int iterate(int m, int n, int53_t (*a)[m], double p) {
    /* Reduce Euclidean norms via projections until they stop changing.

    Parameters
    ----------
    m -- second dimension of a
    n -- first dimension of a
    a -- two-dimensional array whose rows get reduced
    p -- power of the Euclidean norms summed in the objective func. minimized

    Returns
    -------
    number of iterations (rounds of projections) conducted
    a -- two-dimensional array whose rows got reduced
    */
    double (*g)[n] = malloc(sizeof(double[n][n]));
    double (*h)[n] = malloc(sizeof(double[n][n]));
    int53_t (*c)[n] = malloc(sizeof(int53_t[n][n]));
    double s[n];
    int j, k, it, its;
    double sq, sold;

    /* Initialize the Gram matrix. */
    for (k = 0; k < n; k++)
        for (j = 0; j <= k; j++)
            g[k][j] = cblas_ddot(m, a[k], 1, a[j], 1);
    for (k = 0; k < n; k++)
        for (j = k; j < n; j++)
            g[k][j] = g[j][k];
    /* Run iterations. */
    its = 0;
    it = 0;
    s[it] = 1e300;
    do {
        its++;
        sold = s[it];
        /* Calculate the projection coefficients. */
        for (k = 0; k < n; k++)
            for (j = 0; j < n; j++) {
                if (g[k][k] != 0) {
                    c[k][j] = round(g[k][j] / g[k][k]);
                } else {
                    c[k][j] = 0;
                }
            }
        for (k = 0; k < n; k++)
            c[k][k] = 0;
        /* Sum the p-th powers of the Euclidean norms. */
        for (k = 0; k < n; k++) {
            s[k] = 0;
            for (j = 0; j < n; j++) {
                sq = g[j][j] + c[k][j] * (c[k][j] * g[k][k] - 2 * g[k][j]);
                s[k] += pow(sq, p / 2);
            }
        }
        /* Determine which index minimizes the sum. */
        it = 0;
        for (k = 1; k < n; k++)
            if (s[k] < s[it]) it = k;
        /* Project off the it-th vector. */
        for (k = 0; k < n; k++)
            cblas_daxpy(m, -c[it][k], a[it], 1, a[k], 1);
        /* Update the Gram matrix. */
        for (k = 0; k < n; k++) {
            for (j = 0; j <= k; j++) {
                h[k][j] = g[k][j] + 1. * c[it][j] * c[it][k] * g[it][it];
                h[k][j] -= c[it][j] * g[k][it] + c[it][k] * g[it][j];
            }
        }
        for (k = 0; k < n; k++)
            for (j = k; j < n; j++)
                h[k][j] = h[j][k];
        for (k = 0; k < n; k++)
            for (j = 0; j < n; j++)
                g[k][j] = h[k][j];
    } while (s[it] < sold);

    free(g);
    free(h);
    free(c);

    return its;
}


double mean(int n, double a[n]) {
    /* Calculate the mean of the entries of a.

    Parameters
    ----------
    n -- length of a
    a -- array whose entries are to be averaged

    Returns
    -------
    mean of the entries of a
    */
    int k;
    double avg;

    avg = 0;
    for (k = 0; k < n; k++)
        avg += a[k];
    avg /= n;

    return avg;
}


double stddev(int n, double a[n]) {
    /* Calculate the standard deviation of the entries of a.

    Parameters
    ----------
    n -- length of a
    a -- array whose entries go into the standard deviation

    Returns
    -------
    standard deviation of the entries of a
    */
    int k;
    double avg, s;

    avg = mean(n, a);
    s = 0;
    for (k = 0; k < n; k++)
        s += (a[k] - avg) * (a[k] - avg);
    s /= n;
    s = sqrt(s);

    return s;
}


double minimum(int n, double a[n]) {
    /* Calculate the minimum of the entries of a.

    Parameters
    ----------
    n -- length of a
    a -- array whose entries go into the minimum

    Returns
    -------
    minimum of the entries of a
    */
    int k;
    double min;

    min = a[0];
    if (n > 1) {
        for (k = 1; k < n; k++)
            if (a[k] < min) min = a[k];
    }

    return min;
}


double maximum(int n, double a[n]) {
    /* Calculate the maximum of the entries of a.

    Parameters
    ----------
    n -- length of a
    a -- array whose entries go into the maximum

    Returns
    -------
    maximum of the entries of a
    */
    int k;
    double max;

    max = a[0];
    if (n > 1) {
        for (k = 1; k < n; k++)
            if (a[k] > max) max = a[k];
    }

    return max;
}


double minnorm(int m, int n, int53_t (*a)[m]) {
    /* Calculate the Euclidean norm of the shortest row.

    Parameters
    ----------
    m -- second dimension of a
    n -- first dimension of a
    a -- two-dimensional array whose rows are measured

    Returns
    -------
    Euclidean norm of the shortest row of a
    */
    double s, smin;
    int k;

    smin = 1e300;
    for (k = 0; k < n; k++) {
        s = cblas_ddot(m, a[k], 1, a[k], 1);
        if (s < smin) smin = s;
    }

    return sqrt(smin);
}


double frobnorm(int m, int n, int53_t (*a)[m]) {
    /* Calculate the Frobenius norm (sqrt of the sum of squares of entries).

    Parameters
    ----------
    m -- second dimension of a
    n -- first dimension of a
    a -- two-dimensional array whose rows are measured

    Returns
    -------
    Frobenius norm (the square root of the sum of the squares of entries) of a
    */
    double s;
    int k;

    s = 0;
    for (k = 0; k < n; k++)
        s += cblas_ddot(m, a[k], 1, a[k], 1);

    return sqrt(s);
}


void randombasis(int l, int n, int53_t (*a)[n], int53_t q) {
    /* Form a basis for a lattice at random, of the form (q*Id, 0; R, Id).

    Parameters
    ----------
    l -- dimension of the identity matrix in q*Id for (q*Id, 0; R, Id)
    n -- dimension of a
    a -- two-dimensional array to be (q*Id, 0; R, Id), where R is random
    q -- range of the random values generated

    Returns
    -------
    a -- two-dimensional array that is (q*Id, 0; R, Id), where R is random
    */
    int j, k;
    double one;

    /* Fill the upper left block with q times the identity matrix. */
    for (k = 0; k < l; k++)
        for (j = 0; j < l; j++)
            a[k][j] = 0;
    for (k = 0; k < l; k++)
        a[k][k] = q;
    /* Fill the lower left block with random numbers. */
    one = 1 - 1e-15;
    for (k = l; k < n; k++)
        for (j = 0; j < l; j++)
            a[k][j] = round(q * (one * random() / RAND_MAX - one / 2));
    /* Fill the upper right block with zeros. */
    for (k = 0; k < l; k++)
        for (j = l; j < n; j++)
            a[k][j] = 0;
    /* Fill the lower right block with the identity matrix. */
    for (k = l; k < n; k++)
        for (j = l; j < n; j++)
            a[k][j] = 0;
    for (k = l; k < n; k++)
        a[k][k] = 1;
}


char llliteratest(int m, int n, int53_t (*a)[m], double delta, double p,
                  int *its, double minnorms[3], double frobnorms[3],
                  double *t_lll, double *t_iterate) {
    /* Test lll followed by iterate.

    Parameters
    ----------
    m -- second dimension of a
    n -- first dimension of a
    a -- two-dimensional array whose rows get reduced
    delta -- parameter in the Lovasz criterion (0 < delta < 1)
    p -- power of the Euclidean norms summed in the objective func. minimized

    Returns
    -------
    1 if the LLL criteria are satisfied, 0 if the criteria are not satisfied
    a -- two-dimensional array whose rows got reduced
    its -- number of iterations (rounds of projections) conducted
    minnorms -- Euclidean norm of the shortest row, w/ & w/o LLL, & iterate too
    frobnorms -- Frobenius norm of a, with and without LLL, and iterate too
    t_lll -- time in seconds taken by lll
    t_iterate -- time in seconds taken by iterate
    */
    char lllreturn;
    clock_t t0, t1, t01;

    /* Calculate the initial norms. */
    minnorms[0] = minnorm(m, n, a);
    frobnorms[0] = frobnorm(m, n, a);
    /* Run LLL. */
    t0 = clock();
    lll(m, n, a, delta);
    t1 = clock();
    *t_lll = 1. * (t1 - t0) / CLOCKS_PER_SEC;
    lllreturn = lll_check(m, n, a, delta);
    /* Calculate the norms after LLL. */
    minnorms[1] = minnorm(m, n, a);
    frobnorms[1] = frobnorm(m, n, a);
    /* Polish the results with iterations. */
    t0 = clock();
    *its = iterate(m, n, a, p);
    t1 = clock();
    *t_iterate = 1. * (t1 - t0) / CLOCKS_PER_SEC;
    /* Calculate the norms after polishing. */
    minnorms[2] = minnorm(m, n, a);
    frobnorms[2] = frobnorm(m, n, a);

    return lllreturn;
}


int runinstances(int larger, int lesser, int53_t q, unsigned int seed,
                 double delta, double p, int instances,
                 double minnorms[instances][3], double frobnorms[instances][3],
                 double minmulti[3], double frobmulti[3],
                 double t_lll[instances], double t_iterate[instances]) {
    /* Run llliteratetest instances times alone and instances times in series.

    Parameters
    ----------
    larger -- second dimension of R in the test matrix (q*Id, 0; R, Id)
    lesser -- first dimension of R in the test matrix (q*Id, 0; R, Id)
    q -- size of the randomly generated entries of R in (q*Id, 0; R, Id)
    seed -- starting value for the pseudorandom number generator
    delta -- parameter in the Lovasz criterion of the LLL algorithm
    p -- power of the Euclidean norms summed in the objective func. minimized
    instances -- number of examples to run and the length of a series of runs

    Returns
    -------
    0 if the LLL criteria are satisfied, -1 if the criteria are not satisfied
    minnorms -- Euclidean norm of the shortest row, w/ & w/o LLL, & iterate too
    frobnorms -- Frobenius norm, with and without LLL, and iterate too
    minmulti -- same as minnorms, but for a series of instances runs
    frobmulti -- same as frobnorms, but for a series of instances runs
    t_lll -- time in seconds taken by lll when running llliteratest alone
    t_iterate -- time in secs. taken by iterate when running llliteratest alone
    */
    int i, j, k, its, ret;
    double smin, s, minnormi, frobnormi, minnorm1[3], frobnorm1[3];
    double t_lllmulti, t_iteratemulti;

    /* Set the dimensions of (q*Id, 0; R, Id), where R is (lesser x larger). */
    int m = larger + lesser;
    int n = m;
    int53_t (*a)[m] = malloc(sizeof(int53_t[n][m]));

    /* Run instances examples, ordering the rows differently every time. */
    for (k = 0; k < instances; k++) {
        printf("\n");
        printf("instance = %d\n", k + 1);
        /* Seed the random number generator. */
        srandom(seed);
        /* Form a random set of basis vectors. */
        randombasis(larger, n, a, q);
        /* Randomize the rows. */
        for (i = 0; i < k + 1; i++)
            randrows(m, n, a);
        /* Test lll followed by iterate. */
        ret = llliteratest(m, n, a, delta, p, &its, minnorms[k], frobnorms[k],
                           &(t_lll[k]), &(t_iterate[k]));
        if (!ret) {
            printf("lll_check returned 0 for llliteratest run once.\n");
            return -1;
        }
        /* Print the initial norms. */
        printf("minnorm before LLL = %f\n", minnorms[k][0]);
        printf("frobnorm before LLL = %f\n", frobnorms[k][0]);
        /* Print the norms after LLL. */
        printf("minnorm before polishing = %f\n", minnorms[k][1]);
        printf("frobnorm before polishing = %f\n", frobnorms[k][1]);
        /* Print the norms after polishing. */
        printf("minnorm after polishing = %f\n", minnorms[k][2]);
        printf("frobnorm after polishing = %f\n", frobnorms[k][2]);
        printf("its = %d\n", its);
    }

    /* Seed the random number generator. */
    srandom(seed);
    /* Form a random set of basis vectors. */
    randombasis(larger, n, a, q);
    /* Print the initial norms. */
    printf("\n");
    printf("one example processed in succession %d times\n", instances);
    minnormi = minnorm(m, n, a);
    frobnormi = frobnorm(m, n, a);
    printf("minnorm before LLL = %f\n", minnormi);
    printf("frobnorm before LLL = %f\n", frobnormi);
    /* Process the same example instances times in succession. */
    for (k = 0; k < instances; k++) {
        /* Randomize the rows. */
        randrows(m, n, a);
        /* Test lll followed by iterate. */
        ret = llliteratest(m, n, a, delta, p, &its, minnorm1, frobnorm1,
                           &t_lllmulti, &t_iteratemulti);
        if (!ret) {
            printf("lll_check returned 0 for llliteratest run in series.\n");
            return -1;
        }
        /* Print the norms after LLL. */
        printf("minnorm before polishing = %f\n", minnorm1[1]);
        printf("frobnorm before polishing = %f\n", frobnorm1[1]);
        /* Print the norms after polishing. */
        printf("minnorm after polishing = %f\n", minnorm1[2]);
        printf("frobnorm after polishing = %f\n", frobnorm1[2]);
        printf("its = %d\n", its);
    }

    /* Fill the output arrays. */
    minmulti[0] = minnormi;
    minmulti[1] = minnorm1[1];
    minmulti[2] = minnorm1[2];
    frobmulti[0] = frobnormi;
    frobmulti[1] = frobnorm1[1];
    frobmulti[2] = frobnorm1[2];

    free(a);

    return 0;
}


int main(int argc, char *argv[]) {
    /* Set the number of examples to run. */
    const int NUMEX = 7;
    /* Set the number of instances per example to run. */
    const int INSTANCES = 10;
    int i, j, k, large, lesser, larger, ret, x[NUMEX];
    double delta, p, frac[INSTANCES];
    double minnorms[NUMEX][INSTANCES][3], frobnorms[NUMEX][INSTANCES][3];
    double minmean[NUMEX][2], minstddev[NUMEX][2];
    double minmin[NUMEX][2], minmax[NUMEX][2];
    double frobmean[NUMEX][2], frobstddev[NUMEX][2];
    double frobmin[NUMEX][2], frobmax[NUMEX][2];
    double minmulti[NUMEX][3], frobmulti[NUMEX][3];
    double t_lll[NUMEX][INSTANCES], t_iterate[NUMEX][INSTANCES];
    double t_lllavg[NUMEX], t_iterateavg[NUMEX];
    int53_t q;
    FILE *f;
    const int MAXLEN = 100;
    char filename[MAXLEN];
    char stem[MAXLEN];
    char power;

    /* Set large and the parameter delta for the LLL algorithm. */
    if (argc == 2) {
        if (sscanf(argv[1], "%d", &large) == EOF)
            return -2;
        delta = 1 - 1e-15;
    } else if (argc == 3) {
        if (sscanf(argv[1], "%d", &large) == EOF)
            return -2;
        if (sscanf(argv[2], "%lf", &delta) == EOF)
            return -2;
        if (delta < 3e-16 || delta > 1)
            return -2;
    } else {
        printf("There must be either one or two command-line arguments.\n\n");
        printf("The first should be \"1\"%s",
               " for the entries of matrices to be large;\n");
        printf("the first should be \"0\"%s",
               " for the entries of matrices to be smaller.\n\n");
        printf("The second command-line argument is optional to specify.\n");
        printf("When present, the second argument should specify delta\n");
        printf("(delta is the parameter in the Lovasz criterion of LLL).\n");
        printf("As usual, delta must be greater than zero and less than 1.\n");
        printf("With no second argument given, delta defaults to 1-1e-15.\n");
        return -3;
    }
    printf("large = %d\n", large);
    printf("delta = %.16f\n", delta);
    printf("1 - delta = %e\n\n\n\n", 1 - delta);

    /* Set the seed for the random number generator. */
    const unsigned int SEED = 987654321;
    /* Set the stem of the filenames. */
    snprintf(stem, MAXLEN, "plot%d", large);

    /* Set q to be a Mersenne prime, good for the order of a finite field. */
    if (large) {
        q = (1L<<31) - 1;
    } else {
        q = (1L<<13) - 1;
    }

    /* Vary n initially (with power = 0), then vary p (with power = 1). */
    for (power = 0; power <= 1; power++) {
        /* Set the dimensions of q*Id and R for the matrix (q*Id, 0; R, Id). */
        for (k = 0; k < NUMEX; k++) {
            if (k > 0) printf("\n\n\n");
            if (power) {
                lesser = 2 * round(pow(2., NUMEX - 2));
                larger = 2 * lesser;
                p = 1 + k;
                x[k] = p;
            } else {
                lesser = 2 * round(pow(2., k));
                larger = 2 * lesser;
                p = 2;
                x[k] = lesser + larger;
            }
            printf("lesser = %d\n", lesser);
            printf("larger = %d\n", larger);
            printf("p = %f\n", p);
            ret = runinstances(
                larger, lesser, q, SEED, delta, p, INSTANCES,
                minnorms[k], frobnorms[k], minmulti[k], frobmulti[k],
                t_lll[k], t_iterate[k]);
            if (ret) return ret;
            /* Calculate the means, standard deviations, minima, and maxima. */
            t_lllavg[k] = mean(INSTANCES, t_lll[k]);
            t_iterateavg[k] = mean(INSTANCES, t_iterate[k]);
            for (j = 1; j < 3; j++) {
                for (i = 0; i < INSTANCES; i++)
                    frac[i] = minnorms[k][i][j] / minnorms[k][i][j - 1];
                minmean[k][j - 1] = mean(INSTANCES, frac);
                minstddev[k][j - 1] = stddev(INSTANCES, frac);
                minmin[k][j - 1] = minimum(INSTANCES, frac);
                minmax[k][j - 1] = maximum(INSTANCES, frac);
                for (i = 0; i < INSTANCES; i++)
                    frac[i] = frobnorms[k][i][j] / frobnorms[k][i][j - 1];
                frobmean[k][j - 1] = mean(INSTANCES, frac);
                frobstddev[k][j - 1] = stddev(INSTANCES, frac);
                frobmin[k][j - 1] = minimum(INSTANCES, frac);
                frobmax[k][j - 1] = maximum(INSTANCES, frac);
            }
        }

        /* Save the results to disk. */
        for (j = 1; j < 3; j++) {
            /* Save minmean. */
            snprintf(filename, MAXLEN, "%s%d%s%d%s", stem, power, "minmean",
                     j, ".txt");
            f = fopen(filename, "w");
            for (k = 0; k < NUMEX; k++)
                fprintf(f, "%e\n", minmean[k][j - 1]);
            fclose(f);
            /* Save minstddev. */
            snprintf(filename, MAXLEN, "%s%d%s%d%s", stem, power, "minstddev",
                     j, ".txt");
            f = fopen(filename, "w");
            for (k = 0; k < NUMEX; k++)
                fprintf(f, "%e\n", minstddev[k][j - 1]);
            fclose(f);
            /* Save minmin. */
            snprintf(filename, MAXLEN, "%s%d%s%d%s", stem, power, "minmin",
                     j, ".txt");
            f = fopen(filename, "w");
            for (k = 0; k < NUMEX; k++)
                fprintf(f, "%e\n", minmin[k][j - 1]);
            fclose(f);
            /* Save minmax. */
            snprintf(filename, MAXLEN, "%s%d%s%d%s", stem, power, "minmax",
                     j, ".txt");
            f = fopen(filename, "w");
            for (k = 0; k < NUMEX; k++)
                fprintf(f, "%e\n", minmax[k][j - 1]);
            fclose(f);
            /* Save frobmean. */
            snprintf(filename, MAXLEN, "%s%d%s%d%s", stem, power, "frobmean",
                     j, ".txt");
            f = fopen(filename, "w");
            for (k = 0; k < NUMEX; k++)
                fprintf(f, "%e\n", frobmean[k][j - 1]);
            fclose(f);
            /* Save frobstddev. */
            snprintf(filename, MAXLEN, "%s%d%s%d%s", stem, power, "frobstddev",
                     j, ".txt");
            f = fopen(filename, "w");
            for (k = 0; k < NUMEX; k++)
                fprintf(f, "%e\n", frobstddev[k][j - 1]);
            fclose(f);
            /* Save frobmin. */
            snprintf(filename, MAXLEN, "%s%d%s%d%s", stem, power, "frobmin",
                     j, ".txt");
            f = fopen(filename, "w");
            for (k = 0; k < NUMEX; k++)
                fprintf(f, "%e\n", frobmin[k][j - 1]);
            fclose(f);
            /* Save frobmax. */
            snprintf(filename, MAXLEN, "%s%d%s%d%s", stem, power, "frobmax",
                     j, ".txt");
            f = fopen(filename, "w");
            for (k = 0; k < NUMEX; k++)
                fprintf(f, "%e\n", frobmax[k][j - 1]);
            fclose(f);
            /* Save minmulti. */
            snprintf(filename, MAXLEN, "%s%d%s%d%s", stem, power, "minmulti",
                     j, ".txt");
            f = fopen(filename, "w");
            for (k = 0; k < NUMEX; k++)
                fprintf(f, "%e\n", minmulti[k][j] / minmulti[k][j - 1]);
            fclose(f);
            /* Save frobmulti. */
            snprintf(filename, MAXLEN, "%s%d%s%d%s", stem, power, "frobmulti",
                     j, ".txt");
            f = fopen(filename, "w");
            for (k = 0; k < NUMEX; k++)
                fprintf(f, "%e\n", frobmulti[k][j] / frobmulti[k][j - 1]);
            fclose(f);
        }
        /* Save t_lllavg. */
        snprintf(filename, MAXLEN, "%s%d%s%s", stem, power, "t_lll", ".txt");
        f = fopen(filename, "w");
        for (k = 0; k < NUMEX; k++)
            fprintf(f, "%e\n", t_lllavg[k]);
        fclose(f);
        /* Save t_iterateavg. */
        snprintf(filename, MAXLEN, "%s%d%s%s", stem, power, "t_iterate",
                 ".txt");
        f = fopen(filename, "w");
        for (k = 0; k < NUMEX; k++)
            fprintf(f, "%e\n", t_iterateavg[k]);
        fclose(f);
        /* Save x. */
        snprintf(filename, MAXLEN, "%s%d%s%s", stem, power, "x", ".txt");
        f = fopen(filename, "w");
        for (k = 0; k < NUMEX; k++)
            fprintf(f, "%d\n", x[k]);
        fclose(f);
    }

    return 0;
}
