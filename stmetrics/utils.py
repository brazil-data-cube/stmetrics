import numpy


def fixseries(timeseries, nodata=-9999):
    """This function ajusts the time series to polar transformation.

    As some time series may have very significant noises (such as spikes), when
    coverted to polar space it may produce an inconsistent geometry.
    To avoid this issue, this function removes this spikes.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return fixed_timeseries: Numpy array of time series without spikes.
    """
    # check input
    check_input(timeseries)

    # casting to float
    timeseries = timeseries.astype(float)

    # Remove nodata on non masked arrays
    if timeseries[timeseries == nodata].any():
        timeseries[timeseries == nodata] = numpy.nan

    timeseries = timeseries[~numpy.isnan(timeseries)]

    timeseries2 = timeseries.copy()

    idxs = numpy.where(timeseries == 0)[0]

    spikes = list()

    for i in range(len(idxs)-1):
        di = idxs[i+1]-idxs[i]
        if di == 2:
            spikes.append(idxs[i]+1)

    for pos in range(len(spikes)):
        idx = spikes[pos]
        timeseries2[idx] = 0

    return timeseries2


def create_polygon(timeseries):
    """This function converts a time series to the polar space.

    If the time series has lenght smaller than 3, it can not be properly \
    converted to the polar space.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :return polygon: Shapely polygon of time series without spikes.
    """
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import LinearRing

    # remove weird spikes on timeseries
    try:
        ts = fixseries(timeseries)

        list_of_radius, list_of_angles = get_list_of_points(ts)

        # create polygon geometry
        ring = list()

        # add points in the polygon
        N = list_of_radius.shape[0]

        # start to build up polygon
        for i in range(N):
            a = list_of_radius[i] * numpy.cos(2 * numpy.pi * i / N)
            o = list_of_radius[i] * numpy.sin(2 * numpy.pi * i / N)
            ring.append([a, o])

        # Build geometry
        r = LinearRing(ring)

        # Buffer to try make polygon it valid
        polygon = Polygon(r).buffer(0)

        return polygon

    except:
        raise ValueError("Unable to create a valid polygon")


def get_list_of_points(timeseries):
    """This function creates a list of angles based on the time series that \
    is used to convert a time series to a geometry.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :return list_of_observations: Numpy array of lists of observations after \
    polar transformation.

    :return list_of_angles: Numpy array of lists of angles after polar
    transformation.
    """

    list_of_observations = numpy.abs(timeseries)

    list_of_angles = numpy.linspace(0, 2 * numpy.pi, len(list_of_observations))

    return list_of_observations, list_of_angles


def check_input(timeseries):
    """This function checks the input and raises one exception if it is too
    short or has the wrong type.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray.

    :raises ValueError: When ``timeseries`` is not valid.
    """
    dimensions = timeseries.ndim

    if dimensions == 2:
        if timeseries.shape[0] > timeseries.shape[1]:
            dim = 0
        else:
            dim = 1
    elif dimensions == 1:
        dim = 0
    else:
        raise TypeError('Make sure you are using a 2D-array')

    if isinstance(timeseries, numpy.ndarray):
        if timeseries.shape[dim] < 5:
            raise Exception("Your time series is too short!")
        elif numpy.isnan(timeseries).all():
            raise Exception("Your time series has only nans!")
        elif (timeseries == 0).all():
            raise Exception("Your time series has only zeros!")
        else:
            return timeseries
    else:
        raise TypeError('Please use numpy.array as input.')


def file_to_da(filepath):
    import re
    # import pandas
    # import rasterio
    import xarray

    # Open image
    da = xarray.open_rasterio(filepath)
    # transform = da.attrs['transform']

    # find datetime
    match = re.findall(r'\d{4}-\d{2}-\d{2}', filepath)[-1]

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


def bdc2xarray(cube_path, list_bands):
    """This function reads a path with BDC ARD (Brazil Data Cube Analysis
    Ready Data) and creates an xarray dataset.

    :param cube_path: Path of folder with images.
    :type cube_path: string

    :param list_bands: List of bands that will be available as xarray.
    :type list_bands: list

    :return cube_dataset: Xarray dataset.
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
        'sqr_ts': numpy.nan,
        'tqr_ts': numpy.nan
    }
    return basics


def error_polar():
    polares = {
        'area_ts': numpy.nan,
        'angle': numpy.nan,
        'area_q1': numpy.nan,
        'area_q2': numpy.nan,
        'area_q3': numpy.nan,
        'area_q4': numpy.nan,
        'polar_balance': numpy.nan,
        'ecc_metric': numpy.nan,
        'gyration_radius': numpy.nan,
        'csi': numpy.nan
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
    """This function lists the available metrics in stmetrics.
    """
    # import stmetrics
    metrics = [*error_basics().keys(),
               *error_polar().keys(),
               *error_fractal().keys()]

    return metrics


def truncate(n, decimals=6):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier
