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
    