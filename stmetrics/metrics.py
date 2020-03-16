import numpy
from . import basics
from . import polar
from . import fractal

def get_metrics(series,show=False):
    
    #Remove eventual nans from timeseries
    ts = series[~numpy.isnan(series)]
    
    if (not numpy.any(ts)) == True:
        return numpy.zeros((1,20))
        
    #call functions
    basicas = basics.ts_basics(ts)
    polares = polar.ts_polar(ts,show)
    fd = fractal.ts_fractal(ts)

    return numpy.concatenate((basicas, polares,fd), axis=None)

def extractMetrics(series):
    import multiprocessing as mp
    
    #Initialize pool
    pool = mp.Pool(mp.cpu_count())
    
    #use pool to compute metrics for each pixel
    #return a list of arrays
    metrics = pool.map(metrics.get_metrics,[serie for serie in series])
    
    #close pool
    pool.close()    
    
    #Conver list to numpy array
    X_m = numpy.vstack(metrics)
    
    #Concatenate time series and metrics
    X_all = numpy.concatenate((series,X_m), axis=1)
    
    return X_all