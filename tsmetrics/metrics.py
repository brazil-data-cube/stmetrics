import numpy
from . import basics
from . import polar

def get_metrics(series,show=False):

	"""
	This function return all polar and basic metrics in the following order:

	mean, maximum, minimum, std, sum, amplitude, first slope, area, area 1st quarter,area of 2nd quarter,area of 3th quarter, area of 4th quarter, circle, gyration, polar balance e a angle.

	"""
    
    #Remove eventual nans from timeseries
    ts = series[~numpy.isnan(series)]
    
    if (not numpy.any(ts)) == True:
        return numpy.zeros((1,16))
        
    #call functions
    basicas = basics.ts_basics(ts)
    polares = polar.ts_polar(ts,show)

    return numpy.concatenate((basicas, polares), axis=None)
    