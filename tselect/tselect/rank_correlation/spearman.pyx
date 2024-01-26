#cython: language_level=3
import numpy as np

cimport cython
from libc.stdint cimport int64_t



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void spearman_distinct_ranks_cdef(int64_t[:, :] ranks1, int64_t[:, :] ranks2, int n, int nb_classes,
                                       double[:] ocorr) noexcept nogil:
    cdef int i, j
    for i in xrange(nb_classes):
        s = 0.0
        for j in xrange(n):
            s += (ranks1[j, i] - ranks2[j, i])**2
        ocorr[i] = 1 - ((6 * s) / (n * (n**2 - 1)))
#
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void spearman_cdef(int64_t[:, :] ranks1, int64_t[:, :] ranks2, double[:] sd1, double[:] sd2, double[:] cov,
double[:] ocorr) noexcept nogil:
    cdef int i
    for i in xrange(len(sd1)):
        ocorr[i] = cov[i] / (sd1[i] * sd2[i])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void covariance_cdef(int64_t[:, :] ranks1, int64_t[:, :] ranks2, int n, int nb_classes, double[:] cov) noexcept nogil:
    cdef int i, j1, j2
    for i in xrange(nb_classes):
        cov_i = 0.0
        for j1 in xrange(n):
            for j2 in xrange(j1, n):
                cov_i += (ranks1[j1, i] - ranks1[j2, i]) * (ranks2[j1, i] - ranks2[j2, i])
        cov[i] = cov_i / (n**2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void standard_deviation_cdef(int64_t[:, :] ranks1, int n, int nb_classes, double[:] sd) noexcept nogil:
    cdef int i, j1
    for i in xrange(nb_classes):
        mean = 0.0
        sd_i = 0.0

        # compute mean
        for j1 in xrange(n):
            mean += ranks1[j1, i]
        mean = mean/ n

        #compute standard deviation
        for j1 in xrange(n):
            sd_i += (ranks1[j1, i] - mean) ** 2
        sd[i] = (sd_i / n) ** 0.5


#- ENTRY POINTS ---------------------------------------------------------------
def spearman_distinct_ranks(ranks1, ranks2):
    nb_classes = np.shape(ranks1)[1]
    n = np.shape(ranks1)[0]
    if np.shape(ranks2)[0] != n:
        raise ValueError(f"Number of instances do not match: {np.shape(ranks1)} vs {np.shape(ranks2)}")
    if np.shape(ranks2)[1] != nb_classes:
        raise ValueError(f"Number of classes do not match: {np.shape(ranks1)} vs {np.shape(ranks2)}")
    corr = np.zeros(np.shape(ranks1)[1], dtype=np.float64)
    spearman_distinct_ranks_cdef(ranks1, ranks2, n, nb_classes, corr)
    return corr


def spearman(ranks1, ranks2, sd1, sd2):
    nb_classes = np.shape(ranks1)[1]
    n = np.shape(ranks1)[0]
    if np.shape(ranks2)[0] != n:
        raise ValueError(f"Number of instances do not match: {np.shape(ranks1)} vs {np.shape(ranks2)}")
    if np.shape(ranks2)[1] != nb_classes:
        raise ValueError(f"Number of classes do not match: {np.shape(ranks1)} vs {np.shape(ranks2)}")

    cov = np.zeros(nb_classes, dtype=np.float64)
    covariance_cdef(ranks1, ranks2, n, nb_classes, cov)

    corr = np.zeros(nb_classes, dtype=np.float64)
    spearman_cdef(ranks1, ranks2, sd1, sd2, cov, corr)
    return corr

def standard_deviation(ranks1):
    nb_classes = np.shape(ranks1)[1]
    n = np.shape(ranks1)[0]
    sd = np.zeros(nb_classes, dtype=np.float64)
    standard_deviation_cdef(ranks1, n, nb_classes, sd)
    return sd


# def _validate_input(refset, test):
#     if refset.dtype != np.uint8 or test.dtype != np.uint8:
#         raise ValueError("must be byte matrixes")
#     if refset.shape[1] != test.shape[1]:
#         raise ValueError("shapes do not match")
#     if refset.shape[1] >= 255:
#         raise ValueError("possible overflow due to too many trees and use of uint8")
#
# def _prep_input(refset, test, order, dtype):
#     sh = refset.shape
#     sz = dtype().itemsize
#     if sh[1] % sz != 0:
#         z = np.zeros((sh[0], (sh[1]//sz+1)*sz-sh[1]), dtype=np.uint8)
#         refset = np.concatenate((refset, z), axis=1)
#         z = np.zeros((test.shape[0], (sh[1]//sz+1)*sz-sh[1]), dtype=np.uint8)
#         test = np.concatenate((test, z), axis=1)
#     refset = np.array(refset.view(dtype), order=order)
#     test = np.array(test.view(dtype), order=order)
#     return refset, test
#
# def ocscores(refset, test):
#     _validate_input(refset, test)
#     refset, test = _prep_input(refset, test, order="F", dtype=np.uint8)
#
#     out = np.zeros(test.shape[0], dtype=np.uint32)
#
#     _ocscores_colmaj(refset, test, out)
#     return out
#
# def ocscores_topk(refset, test, k):
#     _validate_input(refset, test)
#     refset, test = _prep_input(refset, test, order="F", dtype=np.uint8)
#
#     indexes = np.zeros((test.shape[0], k), dtype=np.uint32)
#     scores = np.zeros((test.shape[0], k), dtype=np.uint8)
#
#     _ocscores_colmaj_topk(refset, test, indexes, scores)
#     return indexes, scores
