def get_metrics(series,show=False):
    import numpy
    from . import basics
    from . import polar
    from . import fractal

    #Remove eventual nans from timeseries
    ts = series[~numpy.isnan(series)]
    
    if (not numpy.any(ts)) == True:
        return numpy.zeros((1,20))
        
    #call functions
    basicas = basics.ts_basics(ts)
    polares = polar.ts_polar(ts,show)
    fd = fractal.ts_fractal(ts)

    return numpy.concatenate((basicas, polares,fd), axis=None)

def sits2metrics(image,merge = False):
    import multiprocessing as mp
    import numpy

    '''
    This function performs the computation of the metrics using multiprocessing.

    Keyword arguments:
    image : numpy.array
        Array of time series. (Series  x Time)
    merge : Boolean
        Indicate if the matrix of features should be merged with the input matrix.
    Returns
    -------
    image : numpy.array
        Numpy matrix of metrics and/or image.

    '''

    # Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (image.shape[1] * image.shape[2], image.shape[0])

    series = image[:,:,:].T.reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(o=image.shape,n=img_as_array.shape))

    #Initialize pool
    pool = mp.Pool(mp.cpu_count())
        
    #use pool to compute metrics for each pixel
    #return a list of arrays
    metricas = pool.map(metrics.get_metrics,[serie for serie in series])
        
    #close pool
    pool.close()    
        
    #Conver list to numpy array
    X_m = numpy.vstack(metricas)

    # Reshape to image shape
    ma = [numpy.reshape(X_m[:,b], image[0,:,:].shape, order='F') for b in range(X_m.shape[1])]
    im_metrics = numpy.rollaxis(numpy.dstack(ma),2)
        
    if merge==True:
        #Concatenate time series and metrics
        X_all = numpy.concatenate((image,im_metrics), axis=0).shape     
        
        return X_all
    else:
        return features