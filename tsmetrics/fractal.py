import numpy
from numba import jit
from math import log, floor
import nolds

from . import utils

all = ['petrosian_fd', 'katz_fd', 'higuchi_fd', 'detrended_fluctuation']

def dfa(series):
    """Detrended Fluctuation Analysis (DFA)
    Parameters
    ----------
    series : list or numpy.array
        One dimensional time series.
    Returns
    -------
    dfa : float
        Detrended Fluctuation Analysis.
    
    This functions uses the dfa implementation from the Nolds package.
    DFA measures the Hurst parameter H, which is very similar to the Hurst exponent. 
    The main difference is that DFA can be used for non-stationary processes (whose mean and/or variance change over time).
    """

    dfa = nolds.dfa(series)
    return dfa

def hurst(series):
    """
    Hurst exponent
    
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

    h = nolds.hurst(series)
    return h

def petrosian_fd(series):
    """Petrosian fractal dimension.
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
    Original code from the `pyrem <https://github.com/gilestrolab/pyrem>`_
    package by Quentin Geissmann.
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
    n = len(x)
    # Number of sign changes in the first derivative of the signal
    diff = numpy.ediff1d(x)
    N_delta = (diff[1:-1] * diff[0:-2] < 0).sum()
    return numpy.log10(n) / (numpy.log10(n) + numpy.log10(n / (n + 0.4 * N_delta)))


def katz_fd(series):
    """Katz Fractal Dimension.
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
    Original code from the `mne-features <https://mne.tools/mne-features/>`_
    package by Jean-Baptiste Schiratti and Alexandre Gramfort.
    References
    ----------
    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
           dimension algorithms. IEEE Transactions on Circuits and Systems I:
           Fundamental Theory and Applications, 48(2), 177-183.
    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
           the computation of EEG biomarkers for dementia." 2nd International
           Conference on Computational Intelligence in Medicine and Healthcare
           (CIMED2005). 2005.
    Examples
    --------
    >>> import numpy as np
    >>> from entropy import katz_fd
    >>> numpy.random.seed(123)
    >>> x = numpy.random.rand(100)
    >>> print(katz_fd(x))
    5.121395665678078
    """
    x = numpy.array(x)
    dists = numpy.abs(numpy.ediff1d(x))
    ll = dists.sum()
    ln = numpy.log10(numpy.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = numpy.max(numpy.abs(aux_d[1:]))
    return numpy.divide(ln, numpy.add(ln, numpy.log10(numpy.divide(d, ll))))


def _higuchi_fd(series, kmax):
    """Utility function for `higuchi_fd`.
    """
    n_times = x.size
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
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
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
    higuchi, _ = utils._linear_regression(x_reg, y_reg)
    return higuchi


def higuchi_fd(series, kmax=10):
    """Higuchi Fractal Dimension.
    Parameters
    ----------
    x : list or numpy.array
        One dimensional time series.
    kmax : int
        Maximum delay/offset (in number of samples).
    Returns
    -------
    hfd : float
        Higuchi fractal dimension.
    Notes
    -----
    Original code from the `mne-features <https://mne.tools/mne-features/>`_
    package by Jean-Baptiste Schiratti and Alexandre Gramfort.
    This function uses Numba to speed up the computation.
    References
    ----------
    .. [1] Higuchi, Tomoyuki. "Approach to an irregular time series on the
       basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2
       (1988): 277-283.
    Examples
    --------
    >>> import numpy as np
    >>> from entropy import higuchi_fd
    >>> numpy.random.seed(123)
    >>> x = numpy.random.rand(100)
    >>> print(higuchi_fd(x))
    2.0511793572134467
    """
    x = numpy.asarray(x, dtype=numpy.float64)
    kmax = int(kmax)
    return _higuchi_fd(x, kmax)