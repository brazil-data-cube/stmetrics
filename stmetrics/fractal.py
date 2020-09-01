import numpy
import nolds
from scipy.signal import savgol_filter

from .utils import *

def ts_fractal(timeseries, funcs=['all'],nodata=-9999):
    
    """
    This function compute 4 fractal dimensions and the hurst exponential.
    
    DFA: measures the Hurst parameter H, which is very similar to the Hurst exponent.
    HE: self-similarity measure that assess long-range dependence in a time series.
    KFD: This algorirhm computes the FD using Katz algorithm.
    PFD: This algorirhm computes the FD of a signal by translating the series into a binary sequence.
    
    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
        numpy.array:
            array of fractal metrics values
    """
        
    metrics_count = 4
    out_metrics = dict()
    
    #Fiz series
    ts = fixseries(timeseries)
    
    if "all" in funcs:
        funcs=['dfa_fd',
        'hurst_exp',
        'katz_fd',
        'entropy']
    
    for f in funcs:
        try:
            out_metrics[f] = eval(f)(ts)
        except:
            print("Sorry, we dont have ", f)
    
    return out_metrics

def dfa_fd(timeseries):
    """
    Detrended Fluctuation Analysis (DFA)

    DFA measures the Hurst parameter H, which is very similar to the Hurst exponent. 
    The main difference is that DFA can be used for non-stationary time series (whose mean and/or variance change over time).

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

    ts = fixseries(timeseries)

    if len(ts)<5:
        interp = savgol_filter(ts,3,2)
    else:
        interp = savgol_filter(ts,5,2)

    try:
        return  nolds.dfa(interp)
    except:
        return numpy.nan

def hurst_exp(timeseries):
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
    
     Notes
    -----

    This function was adapted from the package Nolds.
    """
    ts = fixseries(timeseries)
    
    if len(ts)<5:
        interp = savgol_filter(ts,3,2)
    else:
        interp = savgol_filter(ts,5,2)

    try:
        return nolds.hurst_rs(interp)
    except:
        return numpy.nan

def katz_fd(timeseries):
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
    is the `Euclidean distance <https://en.wikipedia.org/wiki/Euclidean_distance>`_
    between the first point in the series and the point that provides the
    furthest distance with respect to the first point.

    This function was adapted from the package entropy available 
    at: https://github.com/raphaelvallat/entropy.

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
    ts = fixseries(timeseries)
    
    if len(ts)<5:
        interp = savgol_filter(ts,3,2)
    else:
        interp = savgol_filter(ts,5,2)

    try:
        # compute the absolute differences between consecutive elements of an array
        dists = numpy.abs(numpy.ediff1d(interp))
        # sum distances
        l = dists.sum()
        # compute ln using the accumulated distance and the average distance
        ln = numpy.log10(numpy.divide(l, dists.mean()))
        # define box limit 
        d = numpy.max(interp) - numpy.min(interp) 
        
        #return katz fractal dimension
        return numpy.divide(ln, numpy.add(ln, numpy.log10(numpy.divide(d, l))))
    except:
        return numpy.nan

#generalized entropy
def _entropy(series,delta,q):
    v = []
    
    for i in numpy.arange( 1,numpy.ceil((series.max()-series.min())/delta)+1   ):
        b = series[series>(i-1)*delta]
        b = b[b < i*delta]
        v.append(len(b))
        
    v = numpy.array(v)
    
    pi = v/v.sum()
    
    sq = (1/(q-1)) * (1 - (pi**q).sum())
    
    return sq

def entropy(series, delta1=0.1, delta2=0.1, q=0.1):

    sy = _entropy(series, delta1, q)

    dsy = _entropy(numpy.diff(series), delta2, q)

    return sy + dsy + (1-q)*sy*dsy