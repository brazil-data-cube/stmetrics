import numpy

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
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 


    Keyword arguments:
        timeseries : numpy array
            Your time series.

    Returns
    -------
    numpy.array:
        array of basic metrics values

    
    """
    
    #Define header for basic metrics
    #header_basics=["Mean", "Max", "Min", "Std", "Sum","Amplitude","First_slope","Angle"]    
    
    # compute mean, maximum, minimum, standart deviation and amplitude    
    ts = fixseries(timeseries)
    mean = mean_ts(ts)
    maxi = max_ts(ts)
    mini = min_ts(ts)
    std = std_ts(ts)
    soma = sum_ts(ts)
    amp = amplitude(ts)
    slope = first_slop(ts)
    
    return numpy.array([mean,maxi,mini,std,soma,amp,slope])


def mean_ts(timeseries):

    """
    "Mean" - Average value of the curve along one cycle.

    Keyword arguments:
        timeseries : numpy array
            Your time series.

    Returns
    -------
    numpy.float64:
	Mean value of time series.

    """
    
    return numpy.mean(timeseries)
    
def max_ts(timeseries):

    """
    "Max" - Maximum value of the cycle.

    Keyword arguments:
        timeseries : numpy array
            Your time series.

    Returns
    -------
    numpy.float64:
    	Maximum value of time series.
    """
    
    return numpy.max(timeseries)

def min_ts(timeseries):

    """
    "Min" - Minimum value of the curve along one cycle.

    Keyword arguments:
        timeseries : numpy array
            Your time series.

    Returns
    -------
    numpy.float64:
    	Minimum value of time series.
    """
    
    return numpy.min(timeseries)

def std_ts(timeseries):
    """
    "Std" - Standard deviation of the cycle’s values. 

    Keyword arguments:
        timeseries : numpy array
            Your time series.

    Returns
    -------
    numpy.float64:
    	Standard deviation of time series.
    """
    
    return numpy.std(timeseries)

def sum_ts(timeseries):

    """
    "Sum" - Sum of values over a cycle. 
    Usually is an indicator of the annual production of vegetation.

    Keyword arguments:
        timeseries : numpy array
            Your time series.

    Returns
    -------
    numpy.float64:
    	Sum of values of time series.
    """
    
    return numpy.sum(timeseries)
    
def amplitude(timeseries):

    """
    "Amplitude" - The difference between the cycle’s maximum and minimum values.

    Keyword arguments:
        timeseries : numpy array
            Your time series.

    Returns
    -------
    numpy.float64:
    	Amplitude of values of time series.
    """
    
    return numpy.max(timeseries) - numpy.min(timeseries)
    
def first_slop(timeseries):

    """
    "First_slope" - Maximum value of the first slope of the cycle.
    It indicates when the cycle presents some abrupt change in the curve.

    Keyword arguments:
        timeseries : numpy array
            Your time series.

    Returns
    -------
    numpy.float64:
    	The maximum value of the first slope of time series.
    """
    
    return numpy.max(abs(numpy.diff(ts)))

    