import numpy
import warnings
from shapely import geometry
from shapely.geometry.polygon import LinearRing
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
warnings.filterwarnings("ignore")


def fixseries(time_series, nodata=-9999):
    """This function fix the time series.

    As some time series may have very significant noises. When coverted to \
    polar space it may produce inconsistent polygons. To avoid this, this \
    function remove this spikes.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return fixed_timeseries: Numpy array of time series without spikes.
    :rtype fixed_timeseries: numpy.ndarray
    """
    check_input(time_series)

    # Remove nodata on non masked arrays
    if any(time_series[time_series == nodata]):
        time_series[time_series == nodata] = numpy.nan
    
    time_series = time_series[~numpy.isnan(time_series)]

    time_series2 = time_series.copy()

    idxs = numpy.where(time_series == 0)[0]

    spikes = list()
    
    for i in range(len(idxs)-1):
        di = idxs[i+1]-idxs[i]
        if di == 2:
            spikes.append(idxs[i]+1)

    for pos in range(len(spikes)):
        idx = spikes[pos]
        time_series2[idx] = 0

    return time_series2


def create_polygon(timeseries):
    """This function converts the time series to the polar space.
    If the time series has lenght smaller than 3, it cannot be properly\
    converted to polar space.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :return polygon: Numpy array of time series without spikes.
    :rtype polygon: shapely polygon
    """
    # remove weird spikes on timeseries
    try:
        ts = fixseries(timeseries)

        if ts.size == numpy.ones((1,)).size:
            return numpy.array([1])

        list_of_radius, list_of_angles = get_list_of_points(ts)

        ring = list()           # create polygon geometry

        N = len(list_of_radius)         # add points in the polygon

        # start to build up polygon
        for i in range(N):
            a = list_of_radius[i] * numpy.cos(list_of_angles[i])
            o = list_of_radius[i] * numpy.sin(list_of_angles[i])
            ring.append([a, o])
        r = LinearRing(ring)
    
        polygon = Polygon(r).buffer(0)
        return polygon
    
    except:
        print("Unable to create a valid polygon")
        return None
    


def get_list_of_points(ts):
    """This function creates a list of angles based on the time series.
    This list is used for convert the time series to a polygon.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :return list_of_observations: Numpy array of lists of observations after \
    polar transformation.
    :r type list_of_observations: numpy.ndarray

    :return list_of_angles: Numpy array of lists of angles after polar \
    transformation.
    :rtype list_of_observations: numpy.ndarray
    """

    list_of_observations = abs(ts)

    list_of_angles = numpy.linspace(0, 2 * numpy.pi, len(list_of_observations))

    return list_of_observations, list_of_angles


def check_input(timeseries):
    """This function check the input and raise exception if it is too short\
    or has the wrong type.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray.

    :raises ValueError: When ``timeseries`` is not valid.
    """
    if isinstance(timeseries, numpy.ndarray):
        if len(timeseries) < 5:
            raise TypeError("Your time series is too short!")
        else:
            return timeseries
    else:
        raise Exception('Incorrect type: Please use numpy.array as input.')


def file_to_da(filepath):
    import re
    import pandas
    import rasterio
    import xarray

    # Open image
    da = xarray.open_rasterio(filepath)
    transform = da.attrs['transform']

    # find datetime
    match = re.findall(r'\d{4}-\d{2}-\d{2}', filepath)[-1]
    pandas.to_datetime(match)
    da.coords['time'] = match
    return da


def img2xarray(path, band):
    import glob
    import xarray

    # datacube f_path
    f_path = glob.glob(path+"*_"+band+"*.tif")
    f_path.sort()
    dataset = xarray.Dataset()

    # load bands to xarray dataset
    list_of_data_arrays = [file_to_da(link) for link in f_path]

    # load xarray
    dataset[band] = xarray.concat(list_of_data_arrays, dim='time')

    return dataset


def images2xarray(cube_path, list_bands):
    """This function read a path with images and create a xarray dataset.

    :param cube_path: Path of folder with images.
    :type cube_path: string

    :param list_bands: List of bands that will be available on xarray.
    :type list_bands: list

    :return cube_dataset: Xarray dataset.
    :rtype: xarray.dataset
    """
    import xarray

    xray_dataset = [img2xarray(cube_path, band) for band in list_bands]

    cube_dataset = xarray.merge(xray_dataset)

    return cube_dataset


def error_basics():
    basics = {
        'max_ts': numpy.nan,
        'min_ts': numpy.nan,
        'mean_ts': numpy.nan,
        'std_ts': numpy.nan,
        'sum_ts': numpy.nan,
        'amplitude_ts': numpy.nan,
        'mse_ts': numpy.nan,
        'fslope_ts': numpy.nan,
        'skew_ts': numpy.nan,
        'amd_ts': numpy.nan,
        'abs_sum_ts': numpy.nan,
        'iqr_ts': numpy.nan,
        'fqr_ts': numpy.nan,
        'tqr_ts': numpy.nan,
        'sqr_ts': numpy.nan
    }
    return basics


def error_polar():
    polares = {
        'ecc_metric': numpy.nan,
        'gyration_radius': numpy.nan,
        'area_ts': numpy.nan,
        'polar_balance': numpy.nan,
        'angle': numpy.nan,
        'area_q1': numpy.nan,
        'area_q2': numpy.nan,
        'area_q3': numpy.nan,
        'area_q4': numpy.nan,
        'fill_rate': numpy.nan,
        'csi': numpy.nan,
        'fill_rate2': numpy.nan,
        'symmetry_ts': numpy.nan
    }
    return polares


def error_fractal():
    fractais = {
        'dfa_fd': numpy.nan,
        'hurst_exp': numpy.nan,
        'katz_fd': numpy.nan
    }
    return fractais


def list_metrics():
    '''This function list the available metrics in stmetrics.
    '''
    import stmetrics
    metrics = [*error_basics().keys(),
               *error_polar().keys(),
               *error_fractal().keys()]

    return metrics


def truncate(n, decimals=6):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier