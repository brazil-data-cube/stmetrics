import numpy
import warnings
from shapely import geometry
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.polygon import LinearRing

warnings.filterwarnings("ignore")

def fixseries(time_series):
    
    """
    
    This function fix the time series.
    
    As some time series may have very significant noises. When coverted to polar space
    it may produce inconsistent polygons. To avoid this, this function remove this spikes.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    numpy.ndarray:
	   Numpy array of time series without spikes.
    
    """

    check_input(time_series)
    
    if time_series.size == numpy.ones((1,)).size :
        return numpy.array([1])
    
    time_series2 = time_series
    
    idxs = numpy.where(time_series == 0)[0]
    spikes = list()
    for i in range(len(idxs)-1):
        di = idxs[i+1]-idxs[i]
        if di==2:
            spikes.append(idxs[i]+1)  

    for pos in range(len(spikes)):
        idx = spikes[pos]
        time_series2[idx]=0
    
    if len(time_series2) <= 3:
        raise TypeError("Your time series has too much noise we cant compute metrics!")
    
    return time_series2

def create_polygon(timeseries):
    
    """
    
    This function converts the time series to the polar space.
    If the time series has lenght smaller than 3, it cannot be properly converted
    to polar space.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
        Shapely.Polygon
    
    """

    #remove weird spikes on timeseries
    ts = fixseries(timeseries)

    if ts.size == numpy.ones((1,)).size :
        return numpy.array([1])
     
    list_of_radius, list_of_angles = get_list_of_points(ts)

    # create polygon geometry
    ring = list()
    
    # add points in the polygon
    N = len(list_of_radius)
    
    #start to build up polygon
    for i in range(N):
        a = list_of_radius[i] * numpy.cos(2 * numpy.pi * i / N)
        o = list_of_radius[i] * numpy.sin(2 * numpy.pi * i / N)
        ring.append([a,o])
    r = LinearRing(ring)  
    
    try:
        polygon = Polygon(r)
    except:
        print("Unable to create a valid polygon")
    
    return polygon

def get_list_of_points(ts):
    
    """
    
    This function creates a list of angles based on the time series.
    This list is used for convert the time series to a polygon.

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
    lists:
        Lists of observations and angles, that are used to create a polygon.
    
    """
    
    list_of_observations = abs(ts)

    list_of_angles = numpy.linspace(0, 2 * numpy.pi, len(list_of_observations))
    
    return list_of_observations, list_of_angles

def check_input(timeseries):

    """
    
    This function check the inumpyut and raise exception if it is too short or 
    has the wrong type. 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
        timeseries
    
    """
    if timeseries.size == numpy.ones((1,)).size :
        return None

    if isinstance(timeseries,numpy.ndarray):

        if len(timeseries) < 3:
            raise TypeError("Your time series is too short!")
        else:
            return timeseries
    else:
        raise Exception('Incorrect type: Please use numpy.array as input.')

def longlat2window(lon, lat, dataset):
    """
    Args:
        lon (tuple): Tuple of min and max lon
        lat (tuple): Tuple of min and max lat
        dataset: Rasterio dataset

    Returns:
        rasterio.windows.Window
    """
    p = Proj(dataset.crs)
    t = dataset.transform
    xmin, ymin = p(lon[0], lat[0])
    xmax, ymax = p(lon[1], lat[1])
    col_min, row_min = ~t * (xmin, ymin)
    col_max, row_max = ~t * (xmax, ymax)
    return Window.from_slices(rows=(floor(row_max), ceil(row_min)),
                              cols=(floor(col_min), ceil(col_max)))

def file_to_da(filepath):
    import re
    import pandas
    import rasterio
    import xarray
    
    #rasterio.open(1, window=Window(0, 0, 3000, 3000))

    #Open image  
    da = xarray.open_rasterio(filepath)
    transform = da.attrs['transform']
    #find datetime
    match = re.findall(r'\d{4}-\d{2}-\d{2}', filepath)[-1]
    pandas.to_datetime(match)
    da.coords['time'] = match

    return da


def img2xarray(path,band):
    import glob
    import xarray
    
    #datacube f_path
    f_path = glob.glob(path+"*_"+band+"*.tif")   
    f_path.sort()
    dataset = xarray.Dataset()
    
    #load bands to xarray dataset
    list_of_data_arrays=[file_to_da(link) for link in f_path]

    #load xarray
    dataset[band] = xarray.concat(list_of_data_arrays, dim='time')
        
    return dataset


def images2xarray(cube_path,list_bands): 
    import xarray

    xray_dataset = [img2xarray(cube_path,band) for band in list_bands]
    
    cube_dataset = xarray.merge(xray_dataset)

    return cube_dataset
