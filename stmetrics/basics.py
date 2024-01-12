import numpy
from stmetrics.utils import fixseries, truncate


def ts_basics(timeseries, funcs=["all"], nodata=-9999):
    """This function compute all basic metrics in a single call, returning a
    dictionary:

        - "Max" - Maximum value of the time series.

        - "Min" - Minimum value of the time series.

        - "Mean" - Average value of the time series.

        - "Std" - Standard deviation of the time series.

        - "Sum" - Sum of values over a cycle. Usually is an indicator of the \
            annual production of vegetation.

        - "Amplitude" - The difference between the time series’s maximum and \
            minimum values.

        - "MSE" - Mean Spectral Energy.

        - "First_slope" - Maximum value of the first slope of the cycle.

        - "Skew" - Measures the asymmetry of the time series.

        - "AMD" - Absolute mean derivative (AMD).

        - "AbsSum" - Absolute Sum of values over of the time series.

        - "IQR" - Interquaritle range (IQR) of the time series.

        - "FQR" - First quartile of the time series.

        - "SQR" - Second quartile of the time series.

        - "TQR" - Third quaritle of the time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: real number

    :returns: Dictionary of basic metrics
    """

    out_metrics = dict()

    # metrics_count = 15

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
                'sqr_ts',
                'tqr_ts'
                ]

    for f in funcs:
        try:
            out_metrics[f] = eval(f)(timeseries, nodata)
        except:
            out_metrics[f] = numpy.nan

    return out_metrics


def mean_ts(timeseries, nodata=-9999):
    """Average value (mean) of the time series, considering only valid values.
    When nodata is found, it is not included in N value (for all functions).

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Mean value of time series.
    """
    ts = fixseries(timeseries, nodata)

    return truncate(numpy.mean(ts))


def max_ts(timeseries, nodata=-9999):
    """Maximum value of the time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Maximum value of time series.
    """
    ts = fixseries(timeseries, nodata)

    return truncate(numpy.max(ts))


def min_ts(timeseries, nodata=-9999):
    """Minimum value of the time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Minimum value of time series.
    """
    ts = fixseries(timeseries, nodata)

    return truncate(numpy.min(ts))


def std_ts(timeseries, nodata=-9999):
    """Standard deviation of the time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Standard deviation of time series.
    """

    ts = fixseries(timeseries, nodata)

    return truncate(numpy.std(ts))


def sum_ts(timeseries, nodata=-9999):
    """Sum of values of the time series.

    When dealing with vegetation analysis, this value \
    can be a proxy of the annual production of vegetation.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Sum of values of time series.
    """
    ts = fixseries(timeseries, nodata)

    return truncate(numpy.sum(ts))


def amplitude_ts(timeseries, nodata=-9999):
    """The difference between the maximum and minimum \
    values of the time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Amplitude of values of time series.
    """

    ts = fixseries(timeseries, nodata)

    return truncate(numpy.max(ts) - numpy.min(ts))


def fslope_ts(timeseries, nodata=-9999):
    """Maximum value of the first slope of the time series.
    It indicates when some abrupt change happened in the time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The maximum value of the first slope of time series.
    """

    ts = fixseries(timeseries, nodata)

    return truncate(numpy.max(abs(numpy.diff(ts))))


def abs_sum_ts(timeseries, nodata=-9999):
    """Sum of absolute values of the time series. All values are \
    converted to absolute (positive) values, and are summed.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: Sum of absolute values of the time series.
    """

    ts = fixseries(timeseries, nodata)

    return truncate(numpy.sum(numpy.abs(ts)))


def skew_ts(timeseries, nodata=-9999):
    """Measures the asymmetry of the time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The asymmetry of time series.
    """
    from scipy import stats

    ts = fixseries(timeseries, nodata)

    return truncate(stats.skew(ts))


def amd_ts(timeseries, nodata=-9999):
    """Computes the mean of the absolute derivative of time series.

    Regarding to vegetation it provides information on the growth rate of \
    vegetation, allowing discrimination of natural cycles from crop cycles.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The mean of the absolute derivative of time series.
    """
    ts = fixseries(timeseries, nodata)

    return truncate(numpy.mean(numpy.abs(numpy.diff(ts))))


def mse_ts(timeseries, nodata=-9999):
    """Computes mean spectral energy of a time series.

    Mean spectral energy density computes the energy of the time series that \
    is distributed with frequency. High frequency time series usually \
    have lower spectral energy.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The mean spectral energy of the time series.
    """

    ts = fixseries(timeseries, nodata)

    return truncate(numpy.mean(numpy.square(numpy.abs(numpy.fft.fft(ts)))))


def fqr_ts(timeseries, nodata=-9999):
    """Computes the first quartile of a time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The first quartile of the time series.
    """

    ts = fixseries(timeseries, nodata)

    return truncate(numpy.percentile(ts, 25, interpolation='midpoint'))


def tqr_ts(timeseries, nodata=-9999):
    """Computes the third quartile of a time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The third quartile of the time series.
    """

    ts = fixseries(timeseries, nodata)

    return truncate(numpy.percentile(ts, 75, interpolation='midpoint'))


def sqr_ts(timeseries, nodata=-9999):
    """Computes the second quartile of the time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The second quartile of the time series.

    """
    ts = fixseries(timeseries, nodata)

    return truncate(numpy.percentile(ts, 50, interpolation='linear'))


def iqr_ts(timeseries, nodata=-9999):
    """Computes the interquaritle range of the time series.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns: The interquaritle range of the time series.
    """
    ts = fixseries(timeseries, nodata)

    # interpolation is linear by deafult
    q1 = numpy.percentile(ts, 25, interpolation='linear')
    q3 = numpy.percentile(ts, 75, interpolation='linear')

    return truncate(q3 - q1)
