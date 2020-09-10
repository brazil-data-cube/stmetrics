import numpy
import nolds
from scipy.signal import savgol_filter

from . import utils


def ts_fractal(timeseries, funcs=['all'], nodata=-9999):
    """This function compute 4 fractal dimensions and the hurst exponential.

        - DFA: measures the Hurst parameter H, which is very similar to the \
        Hurst exponent.
        - HE: self-similarity measure that assess long-range dependence in a \
        time series.
        - KFD: This algorirhm computes the FD using Katz algorithm.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return out_metrics: Array of fractal metrics values
    :rtype out_metrics: numpy.array
    """
    out_metrics = dict()

    if "all" in funcs:
        funcs = [
                'dfa_fd',
                'hurst_exp',
                'katz_fd'
                ]

    if numpy.all(timeseries == 0) == True:
        out_metrics["fractal"] = utils.error_fractal()
        return out_metrics


    for f in funcs:
        try:
            out_metrics[f] = eval(f)(timeseries)
        except:
            print("Sorry, we had a problem with ", f)

    return out_metrics


def dfa_fd(timeseries):
    """Detrended Fluctuation Analysis (DFA)

    DFA measures the Hurst parameter H, which is very similar to the Hurst \
    exponent.
    The main difference is that DFA can be used for non-stationary time series\
    (whose mean and/or variance change over time).

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :return dfa: Detrended Fluctuation Analysis.
    :rtype dfa: float

    .. Note::
        This functions uses the dfa implementation from the Nolds package.
    """
    ts = utils.fixseries(timeseries)

    interp = savgol_filter(ts, 5, 2)
    return nolds.dfa(interp)
    

def hurst_exp(timeseries):
    """Hurst exponent is a self-similarity measure that assess long-range \
    dependence in a time series. The hurst exponent is a measure of the \
    “long-term memory” of a time series.
    It can be used to determine whether the time series is more, less, or \
    equally likely to increase if it has increased in previous steps.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :return hurst: Hurst expoent.
    :rtype hurst: float

    .. Note::
        This function was adapted from the package Nolds.
    """
    ts = utils.fixseries(timeseries)

    interp = savgol_filter(ts, 5, 2)
    return nolds.hurst_rs(interp)


def katz_fd(timeseries):
    """The Katz fractal dimension is defined by:

    .. math:: K = \\frac{\\log_{10}(n)}{\\log_{10}(d/L)+\\log_{10}(n)}

    where :math:`L` is the total length of the time series and :math:`d` \
    is the Euclidean distancebetween the first point in the series and \
    the point that provides the furthest distance with respect to \
    the first point.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :return kfd: Katz fractal dimension.
    :rtype kfd: float

    .. Note::
        This function was adapted from the package entropy available \
        at: https://github.com/raphaelvallat/entropy.

    .. Tip:: To know more about it:

        Esteller, R. et al. (2001). A comparison of waveform fractal dimension\
        algorithms. IEEE Transactions on Circuits and Systems I: Fundamental \
        Theory and Applications, 48(2), 177-183.

        Goh, Cindy, et al. "Comparison of fractal dimension algorithms for the\
         computation of EEG biomarkers for dementia." 2nd International \
         Conference on Computational Intelligence in Medicine and Healthcare \
         (CIMED2005). 2005.
    """
    ts = utils.fixseries(timeseries)

    interp = savgol_filter(ts, 5, 2)
    # absolute differences between consecutive elements of an array
    dists = numpy.abs(numpy.ediff1d(interp))
    # sum distances
    d_sum = dists.sum()
    # compute ln using the accumulated distance and the average distance
    ln = numpy.log10(numpy.divide(d_sum, dists.mean()))
    # define box limit
    d = numpy.max(interp) - numpy.min(interp)
    ln_sum = numpy.add(ln, numpy.log10(numpy.divide(d, d_sum)))
    # return katz fractal dimension
    return numpy.divide(ln, ln_sum)
