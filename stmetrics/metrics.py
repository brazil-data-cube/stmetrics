import numpy


METRICS_DICT = {
                "basics": ["all"],
                "polar": ["all"],
                "fractal": ["all"]}


def get_metrics(series, metrics_dict=METRICS_DICT, nodata=-9999, show=False):
    """This function performs the computation of the \
    basic, polar and fractal metrics available in the stmetrics package.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param metrics_dict: Dictionary with metrics to be computed.
    :type metrics_dict: dictionary

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns time_metrics: Dicitionary of metrics.
    """
    from .basics import ts_basics
    from .polar import ts_polar
    from .fractal import ts_fractal

    time_metrics = dict()

    # call functions
    if "basics" in metrics_dict:
        time_metrics["basics"] = ts_basics(series,
                                           metrics_dict["basics"],
                                           nodata)

    if "polar" in metrics_dict:
        time_metrics["polar"] = ts_polar(series,
                                         metrics_dict["polar"],
                                         nodata, show)

    if "fractal" in metrics_dict:
        time_metrics["fractal"] = ts_fractal(series,
                                             metrics_dict["fractal"],
                                             nodata)

    return time_metrics


def _getmetrics(timeseries, metrics_dict=METRICS_DICT):

    # Setup empty numpy array
    metricas = numpy.array([])

    # Compute metrics based on dict
    out_metrics = get_metrics(timeseries, metrics_dict, show=False)

    # Loop through dict keys and setup the array
    for key in out_metrics.keys():
        vals = numpy.vstack([out_metrics[key][i]
                             for i in out_metrics[key].keys()])

        # Check size of intial array
        if metricas.size == 0:
            metricas = vals
        else:
            metricas = numpy.vstack((metricas, vals))
    return metricas


def sits2metrics(dataset, metrics=METRICS_DICT, num_cores=-1):
    """This function performs the computation of the metrics using \
    multiprocessing.

    :param dataset: Time series.
    :type dataset: rasterio dataset, numpy array (ZxMxN) - Z \
    is the time series lenght or xarray.Dataset

    :param metrics_dict: Dictionary with metrics to be computed.
    :type metrics_dict: dictionary

    :param num_cores: Number of cores to be used. \
    Value -1 means all cores available.
    :type num_cores: integer \

    :returns image: Numpy matrix of metrics or xarray.Dataset \
    with the metrics as an dataset. The orders of the dimensions, \
    follows the dictionary provided.
    """
    import rasterio
    import xarray

    if isinstance(dataset, rasterio.io.DatasetReader):
        image = dataset.read()
        return _sits2metrics(image, metrics, num_cores)
    elif isinstance(dataset, numpy.ndarray):
        image = dataset.copy()
        return _sits2metrics(image, metrics, num_cores)
    elif isinstance(dataset, xarray.Dataset):
        return _compute_from_xarray(dataset, metrics, num_cores)
    else:
        print("Sorry we can't read this type of file.\
              Please use Rasterio, Numpy array or xarray.")


def _sits2metrics(image, metrics_dict=METRICS_DICT, num_cores=-1):
    import multiprocessing as mp

    # Take our full image, ignore the Fmask band, and reshape into long \
    # 2d array (nrow * ncol, nband) for classification
    new_shape = (image.shape[1] * image.shape[2], image.shape[0])

    # Reshape array
    series = image[:, :, :].T.reshape(new_shape)

    # Check core parameter
    if num_cores == -1:
        num_cores = mp.cpu_count()
    elif num_cores == 0:
        num_cores = 1

    # Initialize pool
    pool = mp.Pool(num_cores)

    # Use pool to compute metrics for each pixel
    # Return a list of arrays
    X_m = pool.starmap(_getmetrics, [(serie.astype(float), metrics_dict)
                                     for serie in series])

    # Close pool
    pool.close()

    # Conver list to numpy array
    metricas = numpy.vstack(X_m)

    # Reshape to image shape
    return metricas.reshape(len(X_m[0]), image.shape[1],
                            image.shape[2], order='F')


def _compute_from_xarray(dataset, metrics=METRICS_DICT, num_cores=-1):
    import xarray
    from .utils import list_metrics

    band_list = list(dataset.data_vars)

    metrics = xarray.Dataset()

    for key in band_list:

        series = numpy.squeeze(dataset[key].values)

        metricas = _sits2metrics(series, metrics, num_cores)

        metrics_list = list_metrics()

        lista = []

        for m, m_name in zip(range(0, metricas.shape[0]), metrics_list):
            c = xarray.DataArray(metricas[m, :, :],
                                 dims=['y', 'x'],
                                 coords={'y': dataset.coords['y'],
                                         'x': dataset.coords['x']})

            c.coords['metric'] = m_name

            lista.append(c)

        dataset[key+'_metrics'] = xarray.concat(lista, dim='metric')

    band_list = None
    lista = None

    return dataset
