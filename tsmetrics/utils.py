import numpy
import math
from numba import jit
from shapely import geometry
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.polygon import LinearRing


def _linear_regression(x, y):
    """Fast linear regression using Numba.
    Parameters
    ----------
    x, y : ndarray, shape (n_times,)
        Variables
    Returns
    -------
    slope : float
        Slope of 1D least-square regression.
    intercept : float
        Intercept
    """
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = numpy.mean(y) - slope * numpy.mean(x)
    return slope, intercept


def _log_n(min_n, max_n, factor):
    """
    Creates a list of values by successively multiplying a minimum value min_n by
    a factor > 1 until a maximum value max_n is reached.
    Non-integer results are rounded down.
    Args:
      min_n (float):
        minimum value (must be < max_n)
    max_n (float):
        maximum value (must be > min_n)
    factor (float):
        factor used to increase min_n (must be > 1)
    Returns:
        list of integers:
            min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
            without duplicates
    """
    assert max_n > min_n
    assert factor > 1
    # stop condition: min * f^x = max
    # => f^x = max/min
    # => x = log(max/min) / log(f)
    max_i = int(numpy.floor(numpy.log(1.0 * max_n / min_n) / numpy.log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(numpy.floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)

    return ns

def fixseries(time_series):
    
    """
    
    This function fix the time series.
    
    As some time series may have very significant noises. When coverted to polar space
    it may produce inconsistent polygons. To avoid this, this function remove this spikes.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.ndarray:
	   Numpy array of time series without spikes.
    
    """

    check_inumpyut(time_series)

    if time_series.size == numpy.ones((1,)).size :
        return numpy.array([1])
    
    time_series2 = time_series
    
    idxs = numpy.where(time_series == 0)[0]
    spikes = list()
    for i in range(len(idxs)-1):
        di = idxs[i+1]-idxs[i]
        if di==2:
            spikes.append(idxs[i]+1)  

    for pos in range(len(spikes)):
        idx = spikes[pos]
        time_series2[idx]=0
    
    if len(time_series2) <= 3:
        raise TypeError("Your time series has too much noise we cant compute metrics!")
    
    return time_series2

def create_polygon(timeseries):
    
    """
    
    This function converts the time series to the polar space.
    If the time series has lenght smaller than 3, it cannot be properly converted
    to polar space.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
        Shapely.Polygon
    
    """
    
    import math

    #remove weird spikes on timeseries
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])
     
    list_of_radius, list_of_angles = get_list_of_points(ts)

    # create polygon geometry
    ring = list()
    
    # add points in the polygon
    N = len(list_of_radius)
    
    #start to build up polygon
    for i in range(N):
        a = list_of_radius[i] * math.cos(2 * numpy.pi * i / N)
        o = list_of_radius[i] * math.sin(2 * numpy.pi * i / N)
        ring.append([a,o])
    r = LinearRing(ring)  
    
    try:
        polygon = Polygon(r)
    except:
        print("Unable to create a valid polygon")
    
    return polygon

def get_list_of_points(ts):
    
    """
    
    This function creates a list of angles based on the time series.
    This list is used for convert the time series to a polygon.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    lists:
        Lists of observations and angles, that are used to create a polygon.
    
    """
    
    list_of_observations = abs(ts)

    list_of_angles = numpy.linspace(0, 2 * numpy.pi, len(list_of_observations))
    
    return list_of_observations, list_of_angles

def check_input(timeseries):

    """
    
    This function check the inumpyut and raise exception if it is too short or 
    has the wrong type. 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
        timeseries
    
    """
    if timeseries.size == numpy.ones((1,)).size :
        return None

    if isinstance(timeseries,numpy.ndarray):

        if len(timeseries) < 3:
            raise TypeError("Your time series is too short!")
        else:
            return timeseries
    else:
        raise Exception('Incorrect type: Please use numpy.array as inumpyut.')


