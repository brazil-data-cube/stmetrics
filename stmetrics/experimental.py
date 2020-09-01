import numpy
from scipy import stats
from stmetrics.utils import *

def ts_exp(timeseries, funcs=["all"], nodata=-9999):
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
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
    numpy.array:
        array of basic metrics values

    
    """
    out_metrics = dict()
    
    metrics_count = 15
    
    # compute mean, maximum, minimum, standart deviation and amplitude    
    ts = fixseries(timeseries)

    if "all" in funcs:
        funcs=[]
    
    for f in funcs:
        try:
            out_metrics[f] = eval(f)(ts,nodata)
        except:
            print("Sorry, we dont have ", f)
    
    return out_metrics


def abs_log(timeseries, nodata=-9999):

    """
    "Mean" - Average value of the curve along one cycle.

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
    numpy.float64:
    Mean value of time series.

    """
    
    ts = fixseries(timeseries,nodata)

    try:
        return numpy.abs(numpy.log(ts))*ts
    except:
        return numpy.nan