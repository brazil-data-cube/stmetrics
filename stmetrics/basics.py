import numpy
from scipy import stats
from .utils import fixseries


def ts_basics(timeseries, funcs=["all"], nodata=-9999):
    """This function compute 7 basic metrics:

        - "Mean" - Average value of the curve along one cycle.
        - "Max" - Maximum value of the cycle.
        - "Min" - Minimum value of the curve along one cycle.
        - "Std" - Standard deviation of the cycle’s values.
        - "Sum" - Sum of values over a cycle. Usually is an indicator of the \
            annual production of vegetation.
        - "Amplitude" - The difference between the cycle’s maximum and \
            minimum values.
        - "First_slope" - Maximum value of the first slope of the cycle.
        - "MSE" - Mean Spectral Energy.
        - "AMD" - Absolute mean derivative (AMD).
        - "Skew" - Measures the asymmetry of the time series.
        - "FQR" - First quartile of the time series.
        - "SQR" - Second quartile of the time series.
        - "TQR" - Third quaritle of the time series.
        - "IQR" - Interquaritle range (IQR) of the time series.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Dictionary of basic metrics
    :rtype: dictionary
    """
    out_metrics = dict()

    metrics_count = 15

    # compute mean, maximum, minimum, standart deviation and amplitude
    ts = fixseries(timeseries)

    if "all" in funcs:
        funcs = [
                'max_ts',
                'min_ts',
                'mean_ts',
                'std_ts',
                'sum_ts',
                'amplitude_ts',
                'mse_ts',
                'fslope_ts',
                'skew_ts',
                'amd_ts',
                'abs_sum_ts',
                'iqr_ts',
                'fqr_ts',
                'tqr_ts',
                'sqr_ts'
                ]

    for f in funcs:
        try:
            out_metrics[f] = eval(f)(ts, nodata)
        except:
            print("Sorry, we had a problem with ", f)

    return out_metrics


def mean_ts(timeseries, nodata=-9999):
    """Mean - Average value of the curve along one cycle.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Mean value of time series.
    :rtype: numpy.float64
    """
    ts = fixseries(timeseries, nodata)

    try:
        return numpy.mean(ts)
    except:
        return numpy.nan


def max_ts(timeseries, nodata=-9999):
    """ Max - Maximum value of the cycle.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Maximum value of time series.
    :rtype: numpy.float64
    """
    ts = fixseries(timeseries, nodata)
    try:
        return numpy.max(ts)
    except:
        return numpy.nan


def min_ts(timeseries, nodata=-9999):
    """Min - Minimum value of the curve along one cycle.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Minimum value of time series.
    :rtype: numpy.float64
    """
    ts = fixseries(timeseries, nodata)

    try:
        return numpy.min(ts)
    except:
        return numpy.nan


def std_ts(timeseries, nodata=-9999):
    """Std - Standard deviation of the cycle’s values.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Standard deviation of time series.
    :rtype: numpy.float64
    """

    ts = fixseries(timeseries, nodata)

    try:
        return numpy.std(ts)
    except:
        return numpy.nan


def sum_ts(timeseries, nodata=-9999):
    """Sum - Sum of values over a cycle.
    Usually is an indicator of the annual production of vegetation.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Sum of values of time series.
    :rtype: numpy.float64
    """
    ts = fixseries(timeseries, nodata)

    try:
        return numpy.sum(ts)
    except:
        return numpy.nan


def amplitude_ts(timeseries, nodata=-9999):
    """Amplitude - The difference between the cycle’s maximum and minimum \
    values.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Amplitude of values of time series.
    :rtype: numpy.float64
    """

    ts = fixseries(timeseries, nodata)

    try:
        return numpy.max(ts) - numpy.min(ts)
    except:
        return numpy.nan


def fslope_ts(timeseries, nodata=-9999):

    """First_slope - Maximum value of the first slope of the cycle.
    It indicates when the cycle presents some abrupt change in the curve.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The maximum value of the first slope of time series.
    :rtype: numpy.float64
    """

    ts = fixseries(timeseries, nodata)

    try:
        return numpy.max(abs(numpy.diff(ts)))
    except:
        return numpy.nan


def abs_sum_ts(timeseries, nodata=-9999):
    """Sum - Sum of values over a cycle.
    Usually is an indicator of the annual production of vegetation.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Sum of values of time series.
    :rtype: numpy.float64
    """

    ts = fixseries(timeseries, nodata)

    try:
        return numpy.sum(numpy.abs(ts))
    except:
        return numpy.nan


def skew_ts(timeseries, nodata=-9999):
    """skew - Measures the asymmetry of the time series.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The asymmetry of time series.
    :rtype: numpy.float64
    """

    ts = fixseries(timeseries, nodata)

    try:
        return stats.skew(ts)
    except:
        return numpy.nan


def amd_ts(timeseries, nodata=-9999):
    """amd - Absolute mean derivative (AMD)
    It provides information on the growth rate of vegetation, allowing \
    discrimination of natural cycles from crop cycles.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The absolute mean derivative of time series.
    :rtype: numpy.float64
    """
    ts = fixseries(timeseries, nodata)

    try:
        return numpy.mean(numpy.abs(numpy.diff(ts)))
    except:
        return numpy.nan


def mse_ts(timeseries, nodata=-9999):
    """mse - Mean Spectral Energy
    It computes mean spectral energy of a time series.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The absolute mean derivative of time series.
    :rtype: numpy.float64

    .. Important::
        This function was adapted from sglearn package.
    """

    ts = fixseries(timeseries, nodata)

    try:
        return numpy.mean(numpy.square(numpy.abs(numpy.fft.fft(ts))))
    except:
        return numpy.nan


def fqr_ts(timeseries, nodata=-9999):
    """fqr - Mean Spectral Energy
    It computes the first quartileof a time series.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The absolute mean derivative of time series.
    :rtype: numpy.float64
    """

    ts = fixseries(timeseries, nodata)

    try:
        return numpy.percentile(ts, 25, interpolation='midpoint')
    except:
        return numpy.nan


def tqr_ts(timeseries, nodata=-9999):
    """tqr - First quartile
    It computes the third quartileof a time series.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The absolute mean derivative of time series.
    :rtype: numpy.float64
    """

    ts = fixseries(timeseries, nodata)

    try:
        return numpy.percentile(ts, 75, interpolation='midpoint')
    except:
        return numpy.nan


def sqr_ts(timeseries, nodata=-9999):
    """sqr - Interquaritle range (IQR)
    It computes the interquaritle range of the time series.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The interquaritle range of the time series.
    :rtype: numpy.float64
    """
    ts = fixseries(timeseries, nodata)

    try:
        return numpy.percentile(ts, 50, interpolation='linear')
    except:
        return numpy.nan


def iqr_ts(timeseries, nodata=-9999):
    """iqr - Interquaritle range (IQR).
    It computes the interquaritle range of the time series.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The interquaritle range of the time series.
    :rtype: numpy.float64
    """
    ts = fixseries(timeseries, nodata)

    if ts.size == numpy.ones((1,)).size:
        return numpy.array([1])

    # interpolation is linear by deafult
    q1 = numpy.percentile(ts, 25, interpolation='linear')
    q3 = numpy.percentile(ts, 75, interpolation='linear')

    try:
        return q3 - q1
    except:
        return numpy.nan
