import numpy

def get_metrics(series, metrics_dict={
        "basics": ["all"],
        "polar": ["all"],
        "fractal": ["all"]
        },
        nodata=-9999, show=False):
    """This function perform the computation and plot of the \
    spectral-polar-fractal metrics available in the stmetrics package.

    :param timeseries: Your time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns time_metrics: Dicitionary of metrics.
    :rtype time_metrics: dictionary
    """
    from . import basics
    from . import polar
    from . import fractal
    from . import utils

    time_metrics = dict()

    # call functions
    if "basics" in metrics_dict:
        time_metrics["basics"] = basics.ts_basics(series, metrics_dict["basics"], nodata)

    if "polar" in metrics_dict:
        time_metrics["polar"] = polar.ts_polar(series, metrics_dict["polar"], nodata, show)
    
    if "fractal" in metrics_dict:
            time_metrics["fractal"] = fractal.ts_fractal(series, metrics_dict["fractal"], nodata)

    return time_metrics


def _getmetrics(timeseries):

    metrics = {
        "basics": ["all"],
        "polar": ["all"],
        "fractal": ["all"]
        }

    out_metrics = get_metrics(timeseries, metrics, show=False)

    metricas = numpy.array([])

    for ki, vi in out_metrics.items():
        for ki2, vi2 in vi.items():
            metricas = numpy.append(metricas,vi2)

    #for metric in out_metrics.keys():
    #    vals = out_metrics[metric].values()
    #    metricas = numpy.append(metricas,
    #        numpy.fromiter(vals, dtype=float), axis=0)
    return metricas


def sits2metrics(dataset):
    """This function performs the computation of the metrics using \
    multiprocessing.

    :param dataset: Your time series.
    :type dataset: rasterio dataset, numpy array (ZxMxN) - Z \
    is the time series lenght or xarray.Dataset

    :returns image: Numpy matrix of metrics or xarray.Dataset \
    with the metrics as an dataset..
    :rtype image: numpy.array or xarray.Dataset.
    """
    import rasterio
    import xarray

    
    if isinstance(dataset, rasterio.io.DatasetReader):
        image = dataset.read()
        return _sits2metrics(image)
    elif isinstance(dataset, numpy.ndarray):
        image = dataset.copy()
        return _sits2metrics(image)
    elif isinstance(dataset, xarray.Dataset):
        return _compute_from_xarray(dataset)
    else:
        print("Sorry we can't read this type of file.\
              Please use Rasterio, Numpy array or xarray.")

def _sits2metrics(image):
    import multiprocessing as mp
    # Take our full image, ignore the Fmask band, and reshape into long \
    # 2d array (nrow * ncol, nband) for classification
    new_shape = (image.shape[1] * image.shape[2], image.shape[0])
    # Reshape array
    series = image[:, :, :].T.reshape(new_shape)
    # Initialize pool
    pool = mp.Pool(mp.cpu_count())
    # use pool to compute metrics for each pixel
    # return a list of arrays
    X_m = pool.map(_getmetrics, [serie for serie in series])
    # close pool
    pool.close()
    # Conver list to numpy array
    metricas = numpy.vstack(X_m)
    # Reshape to image shape
    ma = [numpy.reshape(metricas[:, b], image[0, :, :].shape,
                        order='F') for b in range(metricas.shape[1])]
    im_metrics = numpy.rollaxis(numpy.dstack(ma), 2)

    return im_metrics


def _compute_from_xarray(dataset):

    import xarray
    from . import utils
    
    band_list = list(dataset.data_vars)
    
    metrics = xarray.Dataset()

    for key in band_list:

        series = numpy.squeeze(dataset[key].values)

        metricas = _sits2metrics(series)
        
        metrics_list = utils.list_metrics()
        
        lista = []
        
        for m, m_name in zip(range(0,metricas.shape[0]), metrics_list):
            c = xarray.DataArray(metricas[m,:,:],
                                 dims = ['y','x'],
                                 coords = {'y': dataset.coords['y'],
                                           'x': dataset.coords['x']})
            
            c.coords['metric'] = m_name
                                 
            lista.append(c)
            
        dataset[key+'_metrics'] = xarray.concat(lista, dim='metric')
            
    return dataset
