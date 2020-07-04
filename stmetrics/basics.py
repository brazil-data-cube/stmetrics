import numpy
from scipy import stats
from.utils import *

def ts_basics(timeseries):
    """
    This function compute 7 basic metrics
    "Mean" - Average value of the curve along one cycle.
    "Max" - Maximum value of the cycle.
    "Min" - Minimum value of the curve along one cycle.
    "Std" - Standard deviation of the cycle’s values. 
    "Sum" - Sum of values over a cycle. Usually is an indicator of the annual production of vegetation.
    "Amplitude" - The difference between the cycle’s maximum and minimum values.
    "First_slope" - Maximum value of the first slope of the cycle.
    "mse" - Mean Spectral Energy.
    "amd" - Absolute mean derivative (AMD).
    "skew" - Measures the asymmetry of the time series.
    "fqr" - First quartile of the time series.
    "sqr" - Second quartile of the time series.
    "tqr" - Third quaritle of the time series.
    "iqr" - Interquaritle range (IQR) of the time series.
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 


    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.array:
        array of basic metrics values

    
    """
    
    metrics_count = 14 
    # compute mean, maximum, minimum, standart deviation and amplitude    
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.ones((1,metrics_count))


    means = mean(ts)
    maxi = max(ts)
    mini = min(ts)
    stds = std(ts)
    soma = sum(ts)
    amp = amplitude(ts)
    slope = first_slop(ts)
    skewness = skew(ts)
    amds = amd(ts)
    asum = abs_sum(ts)
    iq = iqr(ts)
    fq = fqr(ts)
    tq = tqr(ts)
    sq = sqr(ts)

    return numpy.array([means,maxi,mini,stds,soma,amp,slope,skewness,amds,asum,fq,sq,tq,iq])


def mean(timeseries):

    """
    "Mean" - Average value of the curve along one cycle.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
	Mean value of time series.

    """
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.mean(ts)
    
def max(timeseries):

    """
    "Max" - Maximum value of the cycle.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
    	Maximum value of time series.
    """
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.max(ts)

def min(timeseries):

    """
    "Min" - Minimum value of the curve along one cycle.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
    	Minimum value of time series.
    """
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.min(ts)

def std(timeseries):
    """
    "Std" - Standard deviation of the cycle’s values. 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
    	Standard deviation of time series.
    """
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.std(ts)

def sum(timeseries):

    """
    "Sum" - Sum of values over a cycle. 
    Usually is an indicator of the annual production of vegetation.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
    	Sum of values of time series.
    """
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.sum(ts)

def amplitude(timeseries):

    """
    "Amplitude" - The difference between the cycle’s maximum and minimum values.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
    	Amplitude of values of time series.
    """
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.max(ts) - numpy.min(ts)
    
def first_slop(timeseries):

    """
    "First_slope" - Maximum value of the first slope of the cycle.
    It indicates when the cycle presents some abrupt change in the curve.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
    	The maximum value of the first slope of time series.
    """
    ts = fixseries(timeseries)
    
    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.max(abs(numpy.diff(ts)))

def abs_sum(timeseries):

    """
    "Sum" - Sum of values over a cycle. 
    Usually is an indicator of the annual production of vegetation.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
        Sum of values of time series.
    """
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.sum(numpy.abs(ts))
    

def skew(timeseries):
    """ 
    "skew" - Measures the asymmetry of the time series

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
        The asymmetry of time series.
    """
    ts = fixseries(timeseries)
    
    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return stats.skew(ts)

def amd(timeseries):
    """ 
    "amd" - Absolute mean derivative (AMD)
    It provides information on the growth rate of vegetation, allowing discrimination of natural cycles from crop cycles.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
        The absolute mean derivative of time series.
    """
    ts = fixseries(timeseries)
    
    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.mean(numpy.abs(numpy.diff(ts)))

def mse(timeseries):
    """ 
    "mse" - Mean Spectral Energy
    It computes mean spectral energy of a time series.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
        The absolute mean derivative of time series.
    
    note: this function was adapted from sglearn package
    """
    ts = fixseries(timeseries)
    
    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.mean(numpy.square(numpy.abs(numpy.fft.fft(ts))))

def fqr(timeseries):
    """ 
    "fqr" - Mean Spectral Energy
    It computes the first quartileof a time series.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
        The absolute mean derivative of time series.
    

    """
    ts = fixseries(timeseries)
    
    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.percentile(ts, 25, interpolation = 'midpoint') 

def tqr(timeseries):
    """ 
    "tqr" - First quartile
    It computes the third quartileof a time series.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
        The absolute mean derivative of time series.
    
    """
    ts = fixseries(timeseries)
    
    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    return numpy.percentile(ts, 75, interpolation = 'linear') 

def sqr(timeseries):
    """ 
    "sqr" - Interquaritle range (IQR) 
    It computes the interquaritle range of the time series.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
        The interquaritle range of the time series.
    
    """
    ts = fixseries(timeseries)
    
    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    #interpolation is linear by deafult
    return numpy.percentile(ts, 50, interpolation = 'linear') 

def iqr(timeseries):
    """ 
    "iqr" - Interquaritle range (IQR) 
    It computes the interquaritle range of the time series.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.float64:
        The interquaritle range of the time series.
    
    """
    ts = fixseries(timeseries)
    
    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])

    #interpolation is linear by deafult
    q1 = numpy.percentile(ts, 25, interpolation = 'linear') 
    q3 = numpy.percentile(ts, 75, interpolation = 'linear') 

    return q3-q1