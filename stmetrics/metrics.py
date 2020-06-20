def get_metrics(series, show=False):
    """
    This function perform the computation and plot of the spectral-polar-fractal metrics available in the stmetrics package.

    Keyword arguments:
    ------------------
        series : numpy.array
            Array of time series.
        show : Boolean
            Indicate if the polar plot should be displayed.
    Returns
    -------
        numpy.array : numpy.array
            Numpy of the metrics.
            
    """
    import numpy
    from . import basics
    from . import polar
    from . import fractal

    #Remove nodata on non masked arrays
    #ts[ts==nodata]=numpy.nan

    #Remove nans from timeseries
    ts = series[~numpy.isnan(series)]

    if numpy.all(ts == 0) == True:
        return numpy.zeros((1,22))
    
    if (not numpy.any(ts)) == True:
        return numpy.zeros((1,22))
        
    #call functions
    basicas = basics.ts_basics(ts)
    polares = polar.ts_polar(ts,show)
    fd = fractal.ts_fractal(ts)

    return numpy.concatenate((basicas, polares,fd), axis=None)

def sits2metrics(dataset,merge = False):
    '''
    This function performs the computation of the metrics using multiprocessing.

    Keyword arguments:
    ------------------
        dataset : rasterio dataset            
        merge : Boolean
            Indicate if the matrix of features should be merged with the input matrix.
    Returns
    -------
        image : numpy.array
            Numpy matrix of metrics and/or image.

    '''

    import multiprocessing as mp
    import numpy

    image = dataset.read()

    # Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (image.shape[1] * image.shape[2], image.shape[0])

    series = image[:,:,:].T.reshape(new_shape)
  
    #Initialize pool
    pool = mp.Pool(mp.cpu_count()-1)
        
    #use pool to compute metrics for each pixel
    #return a list of arrays
    X_m = pool.map(get_metrics,[serie for serie in series])
        
    #close pool
    pool.close()    
        
    #Conver list to numpy array
    metricas = numpy.vstack(X_m)

    # Reshape to image shape
    ma = [numpy.reshape(metricas[:,b], image[0,:,:].shape, order='F') for b in range(metricas.shape[1])]
    im_metrics = numpy.rollaxis(numpy.dstack(ma),2)
        
    if merge==True:
        #Concatenate time series and metrics
        stacked = numpy.concatenate((image,im_metrics), axis=0).shape     
        
        return stacked
    else:
        return im_metrics