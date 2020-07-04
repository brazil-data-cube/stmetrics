import numpy
import nolds

from.utils import *

def ts_fractal(timeseries,kmax=10):
    
    """
    This function compute 4 fractal dimensions and the hurst exponential.
    
    DFA: measures the Hurst parameter H, which is very similar to the Hurst exponent.
    HE: self-similarity measure that assess long-range dependence in a time series.
    KFD: This algorirhm computes the FD using Katz algorithm.
    PFD: This algorirhm computes the FD of a signal by translating the series into a binary sequence.
    
    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
    Returns
    -------
        numpy.array:
            array of fractal metrics values
    """
        
    metrics_count = 3

    #Fiz series
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.ones((1,metrics_count))

    dfa = dfa_fd(ts)
    he = hurst_exp(ts)
    kfd = katz_fd(ts)

    return numpy.array([dfa,he,kfd])

def dfa_fd(series):
    """
    Detrended Fluctuation Analysis (DFA)

    DFA measures the Hurst parameter H, which is very similar to the Hurst exponent. 
    The main difference is that DFA can be used for non-stationary processes (whose mean and/or variance change over time).

    Keyword arguments:
    ------------------
        series : numpy.array
            One dimensional time series.
    Returns
    -------
        dfa : float
            Detrended Fluctuation Analysis.
    
    Notes:
    ------
    This functions uses the dfa implementation from the Nolds package.
    """

    dfa = nolds.dfa(series)
    return dfa

def hurst_exp(series):
    """
    Hurst exponent is a self-similarity measure that assess long-range dependence in a time series.
    
    Keyword arguments:
        series : numpy.array
            One dimensional time series.
    Returns
    -------
        hurst : float
            Hurst exponent.
    
    The hurst exponent is a measure of the “long-term memory” of a time series. 
    It can be used to determine whether the time series is more, less, or equally likely to increase if it has increased in previous steps. 
    This property makes the Hurst exponent especially interesting for the analysis of stock data.
    """

    h = nolds.hurst_rs(series)
    return h

def petrosian_fd(series):
    """
    Petrosian Algorithm.

    This algorirhm computes the FD of a signal by translating the series into a binary sequence.

    Keyword arguments:
    ------------------
        series : numpy.array
            One dimensional time series.
    Returns
    -------
        pfd : float
            Petrosian fractal dimension.

    Notes
    -----
    The Petrosian fractal dimension of a time-series ..:math:`x` is defined by:
    .. math:: P = \\frac{\\log_{10}(N)}{\\log_{10}(N) +
              \\log_{10}(\\frac{N}{N+0.4N_{\\delta}})}
    where ..:math:`N` is the length of the time series, and
    ..:math:`N_{\\delta}` is the number of sign changes in the signal derivative.

    This function was extracted from the package, available at: https://github.com/raphaelvallat/entropy.

    References
    ----------
    .. [1] A. Petrosian, Kolmogorov complexity of finite sequences and
       recognition of different preictal EEG patterns, in , Proceedings of the
       Eighth IEEE Symposium on Computer-Based Medical Systems, 1995,
       pp. 212-217.
    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
       the computation of EEG biomarkers for dementia." 2nd International
       Conference on Computational Intelligence in Medicine and Healthcare
       (CIMED2005). 2005.
    """
    n = len(series)
    # Number of sign changes in the first derivative of the signal
    diff = numpy.ediff1d(series)
    N_delta = (diff[1:-1] * diff[0:-2] < 0).sum()
    return numpy.log10(n) / (numpy.log10(n) + numpy.log10(n / (n + 0.4 * N_delta)))


def katz_fd(series):
    """
    Katz Algorithm.
    
    Keyword arguments:
    ------------------
        series : numpy.array. 
            One dimensional time series.
        
    Returns
    -------
        kfd : float
            Katz fractal dimension.

    Notes
    -----
    The Katz fractal dimension is defined by:

    .. math:: K = \\frac{\\log_{10}(n)}{\\log_{10}(d/L)+\\log_{10}(n)}
    
    where :math:`L` is the total length of the time series and :math:`d`
    is the
    `Euclidean distance <https://en.wikipedia.org/wiki/Euclidean_distance>`_
    between the first point in the series and the point that provides the
    furthest distance with respect to the first point.

    This function was extracted from the package, available at: https://github.com/raphaelvallat/entropy.

    References
    ----------
    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
           dimension algorithms. IEEE Transactions on Circuits and Systems I:
           Fundamental Theory and Applications, 48(2), 177-183.
    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
           the computation of EEG biomarkers for dementia." 2nd International
           Conference on Computational Intelligence in Medicine and Healthcare
           (CIMED2005). 2005.
    """
    x = numpy.array(series)
    dists = numpy.abs(numpy.ediff1d(x))
    ll = dists.sum()
    ln = numpy.log10(numpy.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = numpy.max(numpy.abs(aux_d[1:]))
    return numpy.divide(ln, numpy.add(ln, numpy.log10(numpy.divide(d, ll))))