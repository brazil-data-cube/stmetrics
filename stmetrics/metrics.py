import numpy

def get_metrics(series, metrics_dict={"basics": ["all"],"polar": ["all"],"fractal": ["all"]}, nodata=-9999, show=False):
    """
    This function perform the computation and plot of the spectral-polar-fractal metrics available in the stmetrics package.

    Keyword arguments:
    ------------------
        series : numpy.array
            Array of time series.
        metrics: dictionary
            Dictionary with metrics. Use the following strutcture:
            metrics_dict={"basics": ["all"],
                          "polar": ["all"]
                          "fractal": ["all"]
                          }
        nodata: int/float
            nodata of the time series. Default is -9999.
        show : Boolean
            Indicate if the polar plot should be displayed.
    Returns
    -------
        dictionary : 
            Dicitionary with metrics.
            
    """
    from . import basics
    from . import polar
    from . import fractal
    from . import utils

    time_metrics = dict()

    if numpy.all(series == 0) == True:
        time_metrics["basics"]  = utils.error_basics()
        time_metrics["polar"]   = utils.error_polar()
        time_metrics["fractal"] = utils.error_fractal()
        return time_metrics
    
    if (not numpy.any(series)) == True:
        time_metrics["basics"]  = utils.error_basics()
        time_metrics["polar"]   = utils.error_polar()
        time_metrics["fractal"] = utils.error_fractal()
        return time_metrics
        
    #call functions
    if "basics" in metrics_dict:
        try:
            time_metrics["basics"]  = basics.ts_basics(series, metrics_dict["basics"], nodata)
        except:
            time_metrics["basics"]  = utils.error_basics()

    if "polar" in metrics_dict:
        try:
            time_metrics["polar"]   = polar.ts_polar(series, metrics_dict["polar"], nodata, show)
        except:
            time_metrics["polar"]   = utils.error_polar()

    if "fractal" in metrics_dict:
        try:
            time_metrics["fractal"] = fractal.ts_fractal(series, metrics_dict["fractal"], nodata)
        except:
            time_metrics["fractal"] = utils.error_fractal()
        
    return time_metrics

def _sitsmetrics(timeseries):
    
    metrics = {"basics": ["all"],
           "polar": ["all"],
           "fractal": ["all"]
          }
    
    out_metrics = get_metrics(timeseries, metrics, show=False)
    
    metricas = numpy.array([])

    for metric in out_metrics.keys():
        metricas = numpy.append(metricas, numpy.fromiter(out_metrics[metric].values(), dtype=float), axis = 0)
        
    return metricas

def sits2metrics(dataset,merge = False):
    '''
    This function performs the computation of the metrics using multiprocessing.

    Keyword arguments:
    ------------------
        dataset : rasterio dataset  or numpy array (ZxMxN) - Z is the time series lenght.
        merge : Boolean
            Indicate if the matrix of features should be merged with the input matrix.
    Returns
    -------
        image : numpy.array
            Numpy matrix of metrics and/or image.

    '''

    import multiprocessing as mp
    import rasterio

    if isinstance(dataset, rasterio.io.DatasetReader):
        try:
            image = dataset.read()
            del dataset
        except:
            print('Sorry we could not open your dataset.')
    elif isinstance(dataset, numpy.ndarray): 
        try:
            image = dataset.copy()
            del dataset
        except:
            print('Sorry we could not open your dataset.')
    else:
         print("Sorry we can't read this type of file. Please use Rasterio or Numpy array.")
    
    
    # Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (image.shape[1] * image.shape[2], image.shape[0])

    series = image[:,:,:].T.reshape(new_shape)
  
    #Initialize pool
    pool = mp.Pool(mp.cpu_count())
        
    #use pool to compute metrics for each pixel
    #return a list of arrays
    X_m = pool.map(_sitsmetrics,[serie for serie in series])
        
    #close pool
    pool.close()    

    #Conver list to numpy array
    metricas = numpy.vstack(X_m)

    # Reshape to image shape
    ma = [numpy.reshape(metricas[:,b], image[0,:,:].shape, order='F') for b in range(metricas.shape[1])]
    im_metrics = numpy.rollaxis(numpy.dstack(ma),2)
        
    if merge==True:
        #Concatenate time series and metrics
        return numpy.concatenate((image,im_metrics), axis=0).shape     
    else:
        return im_metrics