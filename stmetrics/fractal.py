import numpy
from numba import jit
from math import log, floor
import nolds

from.utils import *

def dfa_fd(series):
    """Detrended Fluctuation Analysis (DFA)

    DFA measures the Hurst parameter H, which is very similar to the Hurst exponent. 
    The main difference is that DFA can be used for non-stationary processes (whose mean and/or variance change over time).

    Parameters
    ----------
    series : list or numpy.array
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
    Hurst exponent.
    Hurst Exponent is a self-similarity measure that assess long-range dependence in a time series.
    
    Parameters
    ----------
    series : list or numpy.array
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
    """Petrosian Algorithm.

    This algorirhm computes the FD of a signal by translating the series into a binary sequence.

    Parameters
    ----------
    series : list or numpy.array
        One dimensional time series.
    Returns
    -------
    pfd : float
        Petrosian fractal dimension.
    Notes
    -----
    The Petrosian fractal dimension of a time-series :math:`x` is defined by:
    .. math:: P = \\frac{\\log_{10}(N)}{\\log_{10}(N) +
              \\log_{10}(\\frac{N}{N+0.4N_{\\delta}})}
    where :math:`N` is the length of the time series, and
    :math:`N_{\\delta}` is the number of sign changes in the signal derivative.

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
    """Katz Algorithm.
    
    Parameters
    ----------
    x : list or numpy.array
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


def _higuchi_fd(series, kmax):
    """Utility function for `higuchi_fd`.

    Notes
    -----
    This function was extracted from the package, available at: https://github.com/raphaelvallat/entropy.

    """
    n_times = series.size
    lk = numpy.empty(kmax)
    x_reg = numpy.empty(kmax)
    y_reg = numpy.empty(kmax)
    for k in range(1, kmax + 1):
        lm = numpy.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(series[m + j * k] - series[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi


def higuchi_fd(series, kmax=10):
    """Higuchi Fractal Dimension (HFD).

    HFD is defined as the slope of the line that fits the pairs {ln[L(k)],ln(1/k)} in a least-squares sense.  

    where: k indicates the discrete time interval between points.

    Parameters
    ----------
    x : list or numpy.array
        One dimensional time series.
    kmax : int
        Time interval between points. 
    Returns
    -------
    hfd : float
        Higuchi fractal dimension.
    Notes
    -----
    This function was extracted from the package, available at: https://github.com/raphaelvallat/entropy.

    References
    ----------
    .. [1] Higuchi, Tomoyuki. "Approach to an irregular time series on the
       basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2
       (1988): 277-283.
    """
    x = numpy.asarray(series, dtype=numpy.float64)
    kmax = int(kmax)
    return _higuchi_fd(series, kmax)

def ts_fractal(timeseries,kmax=10):
    
    """
    
    This function compute 4 fractal dimensions and the hurst exponential.
    
    DFA: measures the Hurst parameter H, which is very similar to the Hurst exponent. 
    HFD: is defined as the slope of the line that fits the pairs {ln[L(k)],ln(1/k)} in a least-squares sense.  
    HE: self-similarity measure that assess long-range dependence in a time series.
    KFD:
    PFD: This algorirhm computes the FD of a signal by translating the series into a binary sequence.
    
    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
    show: boolean
         This inform that the polar plot must be presented.
    Returns
    -------
    numpy.array:
        array of polar metrics values
    """
    
    #Compute metrics

    ts = fixseries(timeseries)
    
    dfa = dfa_fd(ts)
    hfd = higuchi_fd(ts, kmax=kmax)
    he = hurst_exp(ts)
    kfd = katz_fd(ts)
    pfd = petrosian_fd(ts) 

    return numpy.array([dfa,hfd,he,kfd,pfd])