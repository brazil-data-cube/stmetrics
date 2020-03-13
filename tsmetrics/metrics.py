import numpy
from . import basics
from . import polar

def get_metrics(series,show=False):
    
    #Remove eventual nans from timeseries
    ts = series[~numpy.isnan(series)]
    
    if (not numpy.any(ts)) == True:
        return numpy.zeros((1,16))
        
    #call functions
    basicas = basics.ts_basics(ts)
    polares = polar.ts_polar(ts,show)

    return numpy.concatenate((basicas, polares), axis=None)

def extractMetrics(X):
    from tsmetrics import metrics
    import multiprocessing as mp
    
    #Initialize pool
    pool = mp.Pool(mp.cpu_count())
    
    #use pool to compute metrics for each pixel
    #return a list of arrays
    metrics = pool.map(metrics.get_metrics,[serie for serie in X])
    
    #close pool
    pool.close()    
    
    #Conver list to numpy array
    X_m = numpy.vstack(metrics)
    
    #Concatenate time series and metrics
    X_all = numpy.concatenate((X,X_m), axis=1)
    
    return X_all