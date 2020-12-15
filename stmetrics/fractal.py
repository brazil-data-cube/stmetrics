import numpy
from .utils import fixseries, truncate


def ts_fractal(timeseries, funcs=['all'], nodata=-9999):
    """This function computes 4 fractal dimensions and the hurst exponential.

        - DFA: measures the Hurst parameter H, which is similar to the \
        Hurst exponent.

        - HE: self-similarity measure that assess long-range dependence in a \
        time series.
        
        - KFD: This algorirhm computes the FD using Katz algorithm.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return out_metrics: Dictionary with fractal metrics values.
    """
    out_metrics = dict()

    if "all" in funcs:
        funcs = [
                'dfa_fd',
                'hurst_exp',
                'katz_fd'
                ]

    for f in funcs:
        try:
            out_metrics[f] = eval(f)(timeseries, nodata=nodata)
        except:
            out_metrics[f] = numpy.nan

    return out_metrics


def dfa_fd(timeseries, nvals=None,  overlap=True, order=1, nodata=-9999):
    """Detrended Fluctuation Analysis (DFA) measures the Hurst \
    parameter H, which is very similar to the Hurst Exponent (HE).
    The main difference is that DFA can be used for non-stationary \
    time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nvals: Sizes of subseries to use.
    :type nvals: int

    :param overlap: if True, there will be a 50% overlap on windows \
    otherwise non-overlapping windows will be used.
    :type overlap: Boolean

    :param order: Polynomial order of trend to remove.
    :type order: Boolean

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return dfa: Detrended Fluctuation Analysis.

    .. Note::

        This function uses the Detrended Fluctuation Analysis (DFA) \
        implementation from the Nolds package. Due to time series \
        characteristcs we use by default the 'RANSAC' \
        fitting method as it is more robust to outliers.
        For more details regarding the hurst implementation, check Nolds \
        documentation page.

    """
    import nolds

    ts = fixseries(timeseries, nodata)

    return truncate(nolds.dfa(ts, nvals, overlap, order))


def hurst_exp(timeseries, nvals=None, nodata=-9999):
    """Computes the Hurst Exponent (HE) by a standard \
    rescaled range (R/S) approach.
    HE is a self-similarity measure that assesses long-range \
    dependence in a time series. It can be used to determine whether the \
    time series is more, less, or equally likely to increase if it has \
    increased in previous steps.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray
    
    :param nvals: Sizes of subseries to use.
    :type nvals: int
    
    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return hurst: The Hurst Expoent (HE).

    .. Note::

        This function was adapted from the package Nolds. Due to time series \
        characteristcs we use by default the 'RANSAC' \
        fitting method as it is more robust to outliers.
        For more details regarding the hurst implementation, check Nolds \
        documentation page.
    """
    import nolds
    ts = fixseries(timeseries, nodata)

    return truncate(nolds.hurst_rs(ts, nvals))


def katz_fd(timeseries, nodata=-9999):
    """Katz fractal dimension.

    It is defined by:
    .. math:: K = \\frac{\\log_{10}(n)}{\\log_{10}(d/L)+\\log_{10}(n)}
    where :math:`L` is the total length of the time series and :math:`d` \
    is the Euclidean distance between the first point in the series and \
    the point that provides the furthest distance with respect to \
    the first point.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray
    
    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return kfd: Katz fractal dimension.

    .. Note::
        This function was adapted from the package entropy available \
        at: https://github.com/raphaelvallat/entropy.

    .. Tip:: To know more about it:
        Michael J. Katz, Fractals and the analysis of waveforms, \
        Computers in Biology and Medicine, volume 18, Issue 3,1988,\
        Pages 145-156,ISSN 0010-4825,\
        https://doi.org/10.1016/0010-4825(88)90041-8.
        Esteller, R. et al. (2001). A comparison of waveform fractal dimension\
        algorithms. IEEE Transactions on Circuits and Systems I: Fundamental \
        Theory and Applications, 48(2), 177-183.
        Goh, Cindy, et al. "Comparison of fractal dimension algorithms for the\
        computation of EEG biomarkers for dementia." 2nd International \
        Conference on Computational Intelligence in Medicine and Healthcare \
        (CIMED2005). 2005.
    """
    ts = fixseries(timeseries, nodata)

    # absolute differences between consecutive elements of an array
    dists = numpy.abs(numpy.ediff1d(ts))
    # sum distances
    d_sum = dists.sum()
    # compute ln using the accumulated distance and the average distance
    ln = numpy.log10(numpy.divide(d_sum, dists.mean()))
    # define box limit
    d = numpy.max(ts) - numpy.min(ts)
    ln_sum = numpy.add(ln, numpy.log10(numpy.divide(d, d_sum)))
    # return katz fractal dimension
    return truncate(numpy.divide(ln, ln_sum))
