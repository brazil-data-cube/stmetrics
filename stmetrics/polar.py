import numpy
from shapely import geometry
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.polygon import LinearRing

from .utils import *


def ts_polar(timeseries, funcs=["all"], nodata=-9999, show = False):
    
    """
    
    This function compute 9 polar metrics:
    
    Area - Area of the closed shape.
    Area_q1 - "Area_q4" - Partial area of the shape, proportional to some quadrant of the polar representation
    Eccenticity - Return values close to 0 if the shape is a circle and 1 if the shape is similar to a line.
    Gyration_radius - Equals the average distance between each point inside the shape and the shape’s centroid.
    Polar_balance - The standard deviation of the areas per season, considering the 4 seasons. 
    Angle - The main angle of the closed shape created by the polar visualization.
    
    To visualize the time series on polar space use: ts_polar(ts,show=True)
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
        show: boolean
             This inform that the polar plot must be presented.
    Returns
    -------
    numpy.array:
        array of polar metrics values
    polar plot
    """
    out_metrics = dict()
  
    metrics_count = 9
   
    if "all" in funcs:
        funcs=['ecc_metric',
        'gyration_radius',
        'area_ts',
        'polar_balance',
        'angle',
        'area_q1',
        'area_q2',
        'area_q3',
        'area_q4',
        'fill_rate',
        'shape_index',
        'fill_rate2',
        'symmetry_ts']
    
    for f in funcs:
        try:
            out_metrics[f] = eval(f)(timeseries,nodata)
        except:
            print("Sorry, we dont have ", f)
    
    if show==True:
        polar_plot(timeseries,nodata)
    
    return out_metrics


def symmetric_distance(time_series_1, time_series_2, nodata = -9999):

    """
    
    This function computes the difference between two time series considering the
    polar space.

    Keyword arguments:
    ------------------
        time_series_1 : 
            numpy.ndarray
        time_series_2 : 
            numpy.ndarray    
        nodata: int/float
            nodata of the time series. Default is -9999. 
    Returns
    -------
    numpy.float64:
        distance
    
    """
    #setting up initial distance
    dist = numpy.inf
    pos = numpy.inf
    
    time_series_1[time_series_1==nodata]=numpy.nan
    time_series_1 = time_series_1[~numpy.isnan(time_series_1)]
    
    time_series_2[time_series_2==nodata]=numpy.nan
    time_series_2 = time_series_2[~numpy.isnan(time_series_2)]

    #filtering timeseries
    time_series_1 = fixseries(time_series_1)
    time_series_2 = fixseries(time_series_2)
    #create polygon
    polygon_1 = create_polygon(time_series_1)
    #check validity
    if polygon_1.is_valid == False:
        polygon_1 = polygon_1.buffer(0)
    #Check if one polygon is completly inside other, if not start rolling
    if min(time_series_1) > max(time_series_2) or min(time_series_2) > max(time_series_1):
        polygon_2 = create_polygon(time_series_2)
        polygons_symmetric_difference = polygon_1.symmetric_difference(polygon_2)
        dist = polygons_symmetric_difference.area
    else:
        for i in range(len(time_series_1)):
            shifted_time_series_2 = numpy.roll(time_series_2, i)
            #linalg.norm is efficient way to approximate series
            temp = numpy.linalg.norm(time_series_1-shifted_time_series_2)
            #save the minimum distance
            if temp<dist:
                dist = temp
                pos = i
        #roll time series
        time_series_2 = numpy.roll(time_series_2,pos)
        #create polygon 
        polygon_2 = create_polygon(time_series_2)
        #check validity 
        if polygon_2.is_valid:
            #compute symmetric difference of time series
            polygons_symmetric_difference = polygon_1.symmetric_difference(polygon_2)

            dist = polygons_symmetric_difference.area

        else:

            polygon_2 = polygon_2.buffer(0)

            polygons_symmetric_difference = polygon_1.symmetric_difference(polygon_2)

            dist = polygons_symmetric_difference.area
        
    return dist

def polar_plot(timeseries, nodata=-9999):
    """
    
    This function create a plot of time series in polar space.

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.

    Returns
    -------
        None: Plot of time series in polar space.
    
    """
    import matplotlib.pyplot as plt
    from descartes import PolygonPatch

    #filter time series
    ts = fixseries(timeseries, nodata)
    #create polygon
    polygon = create_polygon(ts)
    #get polygon coords
    x,y = polygon.envelope.exterior.coords.xy
    minX = -numpy.max(numpy.abs(x))
    minY = -numpy.max(numpy.abs(y))
    maxX = numpy.max(numpy.abs(x))
    maxY = numpy.max(numpy.abs(y))
    #get season rings
    ringTopLeft,ringTopRight,ringBottomLeft,ringBottomRight = get_seasons(x,y)
    #setup plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = PolygonPatch(ringTopLeft,fc='#F7CF89', ec=None, alpha=0.5)
    patch2 = PolygonPatch(ringTopRight,fc='#8BF789', ec=None, alpha=0.5)
    patch3 = PolygonPatch(ringBottomLeft,fc='#D09A3C', ec=None, alpha=0.5)
    patch4 = PolygonPatch(ringBottomRight,fc='#6CBEEA', ec=None, alpha=0.5)
    patch5 = PolygonPatch(polygon,fc='#EF8C78', ec=None, alpha=0.5)
    plt.ylim(top=maxY)  # adjust the top
    plt.ylim(bottom=minY)  # adjust the bottom 
    plt.xlim(left=minX)  # adjust the left
    plt.xlim(right=maxX)  # adjust the right
    ax.add_patch(patch)
    ax.add_patch(patch2)
    ax.add_patch(patch3)
    ax.add_patch(patch4)
    ax.add_patch(patch5)
    plt.show()
    
    return None

def get_seasons(x,y):
    
    """
    
    This function polygons that represents the four season of a year.
    They are used to compute the metric "area_season."
    
    Keyword arguments:
    ------------------
        x,y coordinates:
            Using this coordinates is built the quadrants that represents seasons.

    Returns
    -------
        Polygons
    
    """
    
    #get bouding box
    minX = -numpy.max(numpy.abs(x))
    minY = -numpy.max(numpy.abs(y))
    maxX = numpy.max(numpy.abs(x))
    maxY = numpy.max(numpy.abs(y))

    '''
    coord0----coord1----coord2
    |           |           |
    coord3----coord4----coord5
    |           |           |
    coord6----coord7----coord8
    '''

    coord0 = minX, maxY
    coord1 = 0, maxY
    coord2 = maxX, maxY
    coord3 = minX, 0
    coord4 = 0,0
    coord5 = maxX, 0
    coord6 = minX, minY
    coord7 = 0, minY
    coord8 = maxX, minY
    # compute season polygons
    polyTopLeft = Polygon([coord0,coord3,coord4,coord1,coord0])
    polyTopRight = Polygon([coord1,coord2,coord5,coord4,coord1])
    polyBottomLeft = Polygon([coord3,coord4,coord7,coord6,coord3])
    polyBottomRight = Polygon([coord4,coord5,coord8,coord7,coord4])
    
    polyTopLeft = polyTopLeft.buffer(0)
    polyTopRight = polyTopRight.buffer(0)
    polyBottomLeft = polyBottomLeft.buffer(0)
    polyBottomRight = polyBottomRight.buffer(0)

    return polyTopLeft,polyTopRight,polyBottomLeft,polyBottomRight

def area_season(timeseries, nodata=-9999):
    
    """

    Area per season - Partial area of the shape, proportional to some quadrant of the polar representation.
    This metric returns the area of the polygon on each quadrant.
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
    numpy.float64
        The area of the time series that intersected each quadrant that represents a season.
    
    """
    
    #fix time series
    ts = fixseries(timeseries, nodata)

    #create polygon
    try:
        polygon = create_polygon(ts)
        polygon = polygon.buffer(0)
    except:
        return numpy.nan,numpy.nan,numpy.nan,numpy.nan
       
    """
    area2----area1
    |           | 
    area3----area4
    """
    
    #get polygon coords
    x,y = polygon.envelope.exterior.coords.xy
    
    #get season polygons
    polyTopLeft,polyTopRight,polyBottomLeft,polyBottomRight = get_seasons(x,y)

    #compute intersection of season polygons with time series polar representation
    quaterPolyTopLeft = polyTopLeft.intersection(polygon)
    quaterPolyTopRight =  polyTopRight.intersection(polygon)
    quaterPolyBottomLeft =  polyBottomLeft.intersection(polygon)
    quaterPolyBottomRight =  polyBottomRight.intersection(polygon)

    #compute areas
    #area1 =  quaterPolyTopRight.area
    #area2 = quaterPolyTopLeft.area
    #area3 =  quaterPolyBottomLeft.area
    #area4 =  quaterPolyBottomRight.area
    
    return area1,area2,area3,area4

def area_q1(timeseries, nodata=-9999):
    
    """
    Area_Q1 - Area of the closed shape over the first quadrant.
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 
    

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
        numpy.float64: Area of polygon.
        
    """
    
    try:
        areas = area_season(timeseries, nodata)
        return areas[0].area
    except:
        return numpy.nan

    

def area_q2(timeseries, nodata=-9999):
    """
    Area_Q1 - Area of the closed shape over the second quadrant.
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 
    

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
        numpy.float64: Area of polygon.
        
    """

    try:
        areas = area_season(timeseries, nodata)
        return areas[1].area
    except:
        return numpy.nan

def area_q3(timeseries, nodata=-9999):
    """
    Area_Q1 - Area of the closed shape over the thrid quadrant.
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 
    

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
        numpy.float64: Area of polygon.
        
    """
    try:
        areas = area_season(timeseries, nodata)
        return areas[2].area
    except:
        return numpy.nan

def area_q4(timeseries, nodata=-9999):
    """
    Area_Q4 - Area of the closed shape over the last quadrant.
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 
    

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
        numpy.float64: Area of polygon.
        
    """
    try:
        areas = area_season(timeseries, nodata)
        return areas[3].area
    except:
        return numpy.nan

def ecc_metric(timeseries, nodata=-9999):
    
    """

    Eccenticity - Return values close to 0 if the shape is a circle and 1 if the shape is similar to a line.    
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
    numpy.float64
    Eccentricity of time series.
    
    """

    try:
        #filter time series
        ts = fixseries(timeseries, nodata)
        #create polygon
        polygon = create_polygon(ts)     
        #get MRR
        rrec = polygon.minimum_rotated_rectangle
        minx, miny, maxx, maxy = rrec.bounds
        axis1 = maxx - minx
        axis2 = maxy - miny
        stats = numpy.array([axis1, axis2])
        return (stats.min() / stats.max())
    except:
        return numpy.nan

def angle(timeseries, nodata=-9999):
    
    """
    Angle - The main angle of the closed shape created by the polar visualization.
    If two angle are the same, the first one is presented
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
    numpy.float64:
        The main angle of time series.
    
    """

    try:
        #filter time series
        ts = fixseries(timeseries, nodata)
        #get polar transformation info
        list_of_radius, list_of_angles = get_list_of_points(ts)

        return list_of_angles[numpy.argmax(list_of_radius)]
    except:
        return numpy.nan

def gyration_radius(timeseries, nodata=-9999):
    
    """
    Gyration_radius - Equals the average distance between each point inside the shape and the shape’s centroid.
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
    numpy.float64
       Average distance between each point inside the shape and the shape’s centroid.
    
    """

    try:

        #filtered time series
        ts = fixseries(timeseries, nodata)
    
        #create polygon
        polygon = create_polygon(ts)   
        #get polygon centroids
        lonc,latc = polygon.centroid.xy
        #get polygon exterior coords
        x,y = polygon.exterior.coords.xy

        dist = []
        #compute distances to the centroid.
        for p in range(len(x)):
            px = x[p]
            py = y[p]

            dist = numpy.sqrt((px-lonc[0])**2 + (py-latc[0])**2)   

        return numpy.mean(dist)
    except:
        return numpy.nan

def polar_balance(timeseries, nodata=-9999):
    
    """
    Polar_balance - The standard deviation of the areas per season, considering the 4 seasons. 
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
    numpy.float64
       Standard deviation of the areas per season.
    
    """

  
   
    try:
        #filter time series
        ts = fixseries(timeseries, nodata)

        #get area season
        areas = area_season(ts)

        #return polar balance    
        return numpy.std(areas)

    except:
        return numpy.nan


def area_ts(timeseries, nodata=-9999):
    
    """
    Area - Area of the closed shape.
    
    Reference: Körting, Thales & Câmara, Gilberto & Fonseca, Leila. (2013). \\
    Land Cover Detection Using Temporal Features Based On Polar Representation. 
    

    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
    numpy.float64
    Area of polygon.
    
    """

    try:  
        #fix time series
        ts = fixseries(timeseries, nodata)

        #create polygon
        polygon = create_polygon(ts)   

        #return polygon area
        return polygon.area
    except:
        return numpy.nan

def fill_rate(timeseries, nodata = -9999):
    import pointpats
    from shapely.geometry import Point
    
    try:
        #fix time series
        ts = fixseries(timeseries, nodata)
        
        #create polygon
        polygon = create_polygon(ts)   

        #compute convex hull
        convex = polygon.convex_hull 

        return polygon.symmetric_difference(convex).area#   /polygon.area

    except:
        return numpy.nan

def shape_index(timeseries, nodata=-9999):
    """

    Shape index - This is a dimensionless quantitative measure of morphology.
    Characterize the standard deviation of an object from a circle.
    
    Reference: Volkan Müjdat Tiryaki and Usienemnfon Adia-Nimuwa and Virginia M. Ayres and Ijaz Ahmed and David I. Shreiber
    Texture-based segmentation and a new cell shape index for quantitative analysis of cell spreading in AFM images. Cytometry Part A, 2015.
    
    Keyword arguments:
    ------------------
        timeseries : numpy.ndarray
            Your time series.
        nodata: int/float
            nodata of the time series. Default is -9999.
    Returns
    -------
    numpy.float64
        Quantitative measure of morphology.
    
    """
    
    try:
        #filter time series
        ts = fixseries(timeseries, nodata)
        
        #create polygon
        polygon = create_polygon(ts).buffer(0)   
        
        
        #get polar transformation info
        return (polygon.length**2)/(4*numpy.pi*polygon.area)
    except:
        return numpy.nan
    
    
def fill_rate2(timeseries, nodata = -9999):
    import pointpats
    from shapely.geometry import Point   
    
    try:
        #fix time series
        ts = fixseries(timeseries, nodata)
        
        #create polygon
        polygon = create_polygon(ts).buffer(0)
        center = (0,0)
        mbc_poly = Point(*center).buffer(numpy.max(ts))

        return polygon.symmetric_difference(mbc_poly).area#/polygon.area

    except:
        return numpy.nan
    
def symmetry_ts(timeseries, nodata = -9999):
    import pointpats
    from shapely.geometry import Point
    from shapely import affinity
    from shapely.ops import cascaded_union
    
    try: 
        #fix time series
        ts = fixseries(timeseries, nodata)

        #create polygon
        polygon = create_polygon(ts).buffer(0)

        rotated = create_polygon(numpy.roll(ts,int(numpy.ceil(len(ts)/2)))).buffer(0)

        merge = cascaded_union([polygon,rotated]) 

        return  merge.symmetric_difference(polygon).area
    except:
        return numpy.nan   