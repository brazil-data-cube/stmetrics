import numpy
from .utils import fixseries, truncate, create_polygon, get_list_of_points


def ts_polar(timeseries, funcs=["all"], nodata=-9999, show=False):
    """This function compute 9 polar metrics:

    - Area - Area of the closed shape.

    - Angle - The main angle of the closed shape created after transformation.

    - Area_q1 - Partial area of the shape, proportional to quadrant 1 of the \
    polar representation.

    - Area_q2 - Partial area of the shape, proportional to quadrant 2 of the \
    polar representation.

    - Area_q3 - Partial area of the shape, proportional to quadrant 3 of the \
    polar representation.

    - Area_q4 - Partial area of the shape, proportional to quadrant 4 of the \
    polar representation.

    - Polar_balance - The standard deviation of the areas per season,\
    considering the 4 seasons.

    - Eccenticity - Return values close to 0 if the shape is a circle and 1\
    if the shape is similar to a line.

    - Gyration_radius - Equals the average distance between each point inside\
     the shape and the shape’s centroid.

    - CSI - This is a dimensionless quantitative measure of morphology, \
    that characterize the standard deviation of an object from a circle.

    To visualize the time series on polar space use: ts_polar(timeseries,\
     show=True)

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :param show: This inform that the polar plot must be presented.
    :type nodata: boolean

    :returns out_metrics: Dictionary with polar metrics values.

    .. Tip::

        Check the original publication of the metrics: Körting, Thales \
        & Câmara, Gilberto & Fonseca, Leila. (2013). \
        Land Cover Detection Using Temporal Features \
        Based On Polar Representation.
    """
    out_metrics = dict()

    # metrics_count = 9

    if "all" in funcs:
        funcs = [
            'area_ts',
            'angle',
            'area_q1',
            'area_q2',
            'area_q3',
            'area_q4',
            'polar_balance',
            'ecc_metric',
            'gyration_radius',
            'csi'
            ]

    for f in funcs:
        try:
            out_metrics[f] = eval(f)(timeseries, nodata)
        except:
            out_metrics[f] = numpy.nan

    if show is True:
        polar_plot(timeseries, nodata)

    return out_metrics


def symmetric_distance(time_series_1, time_series_2, nodata=-9999):
    """This function computes the difference between two time series \
    in the polar space.

    :param timeseries1: Time series.
    :type timeseries: numpy.ndarray

    :param timeseries2: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns dist: Distance between two time series.
    """
    # setting up initial distance
    dist = numpy.inf
    pos = numpy.inf

    time_series_1[time_series_1 == nodata] = numpy.nan
    time_series_1 = time_series_1[~numpy.isnan(time_series_1)]

    time_series_2[time_series_2 == nodata] = numpy.nan
    time_series_2 = time_series_2[~numpy.isnan(time_series_2)]

    # filtering timeseries
    time_series_1 = fixseries(time_series_1)
    # filtering timeseries
    time_series_2 = fixseries(time_series_2)
    # create polygon
    polygon_1 = create_polygon(time_series_1).buffer(0)

    # Check if one polygon is completly inside other, if not start rolling
    test_1 = min(time_series_1) > max(time_series_2)
    test_2 = min(time_series_2) > max(time_series_1)

    if test_1 or test_2:
        polygon_2 = create_polygon(time_series_2)
        poly_sym_difference = polygon_1.symmetric_difference(polygon_2)
        dist = poly_sym_difference.area
    else:
        for i in range(len(time_series_1)):
            shifted_time_series_2 = numpy.roll(time_series_2, i)

            # linalg.norm is efficient way to approximate series
            temp = numpy.linalg.norm(time_series_1 - shifted_time_series_2)

            # save the minimum distance
            if temp < dist:
                dist = temp
                pos = i

        # roll time series
        time_series_2 = numpy.roll(time_series_2, pos)
        # create polygon
        polygon_2 = create_polygon(time_series_2).buffer(0)

        # compute symmetric difference of time series
        poly_sym_difference = polygon_1.symmetric_difference(polygon_2)
        dist = poly_sym_difference.area

    return truncate(dist)


def polar_plot(timeseries, nodata=-9999):
    """This function create a plot of time series in polar space.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :returns plot: Plot of time series in polar space.
    """
    import matplotlib.pyplot as plt
    from descartes import PolygonPatch
    # from matplotlib.ticker import FormatStrFormatter

    # filter time series
    ts = fixseries(timeseries, nodata)
    # create polygon
    polygon = create_polygon(ts)
    # get polygon coords
    x, y = polygon.envelope.exterior.coords.xy
    minX = -numpy.max(numpy.abs(x))
    minY = -numpy.max(numpy.abs(y))
    maxX = numpy.max(numpy.abs(x))
    maxY = numpy.max(numpy.abs(y))

    # get season rings
    ringTopLeft, ringTopRight, \
        ringBottomLeft, ringBottomRight = get_seasons(x, y)

    # setup plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = PolygonPatch(ringTopLeft, fc='#F7CF89', ec=None, alpha=0.5)
    patch2 = PolygonPatch(ringTopRight, fc='#8BF789', ec=None, alpha=0.5)
    patch3 = PolygonPatch(ringBottomLeft, fc='#D09A3C', ec=None, alpha=0.5)
    patch4 = PolygonPatch(ringBottomRight, fc='#6CBEEA', ec=None, alpha=0.5)
    patch5 = PolygonPatch(polygon, fc='#EF8C78', ec=None, alpha=0.5)
    plt.ylim(top=maxY)          # adjust the top
    plt.ylim(bottom=minY)       # adjust the bottom
    plt.xlim(left=minX)         # adjust the left
    plt.xlim(right=maxX)        # adjust the right
    ax.add_patch(patch)
    ax.add_patch(patch2)
    ax.add_patch(patch3)
    ax.add_patch(patch4)
    ax.add_patch(patch5)
    # adjust ticks
    locs, labels = plt.xticks()
    locsy, labelsy = plt.yticks()
    plt.xticks(locs[1:-1], numpy.round(abs(locs)[1:-1], 4), rotation=90)
    plt.yticks(locsy[1:-1], numpy.round(abs(locs)[1:-1], 4))

    plt.show()

    return None


def get_seasons(x, y):
    """This function polygons that represents the four season of a year.
    They are used to compute the metric ``area_season``.

    :param x: x-coordinate in polar space.
    :type x: numpy.array

    :param y: y-coordinate in polar space.
    :type y: numpy.array

    :returns tuple of polygons: Quadrant polygons
    """
    from shapely.geometry import Polygon

    # get bouding box
    minX = -numpy.max(numpy.abs(x))
    minY = -numpy.max(numpy.abs(y))
    maxX = numpy.max(numpy.abs(x))
    maxY = numpy.max(numpy.abs(y))

    coord0 = minX, maxY
    coord1 = 0, maxY
    coord2 = maxX, maxY
    coord3 = minX, 0
    coord4 = 0, 0
    coord5 = maxX, 0
    coord6 = minX, minY
    coord7 = 0, minY
    coord8 = maxX, minY

    # compute season polygons
    polyTopLeft = Polygon([coord0, coord3, coord4, coord1, coord0])
    polyTopRight = Polygon([coord1, coord2, coord5, coord4, coord1])
    polyBottomLeft = Polygon([coord3, coord4, coord7, coord6, coord3])
    polyBottomRight = Polygon([coord4, coord5, coord8, coord7, coord4])

    polyTopLeft = polyTopLeft.buffer(0)
    polyTopRight = polyTopRight.buffer(0)
    polyBottomLeft = polyBottomLeft.buffer(0)
    polyBottomRight = polyBottomRight.buffer(0)

    return polyTopLeft, polyTopRight, polyBottomLeft, polyBottomRight


def area_season(timeseries, nodata=-9999):
    """Partial area of the shape, proportional \
    to some quadrant of the polar representation.

    This metric returns the area of the polygon on each quadrant.

    area2----area1
    |           |
    area3----area4

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return area: The area of the time series that intersected each \
    quadrant that represents a season.
    """
    # fix time series
    ts = fixseries(timeseries, nodata)
    # create polygon

    polygon = create_polygon(ts)
    polygon = polygon.buffer(0)

    # get polygon coords
    x, y = polygon.envelope.exterior.coords.xy

    # get season polygons
    polyTopLeft, polyTopRight, \
        polyBottomLeft, polyBottomRight = get_seasons(x, y)

    # compute intersection of season polygons with time series polar\
    # representation
    quaterPolyTopLeft = polyTopLeft.intersection(polygon)
    quaterPolyTopRight = polyTopRight.intersection(polygon)
    quaterPolyBottomLeft = polyBottomLeft.intersection(polygon)
    quaterPolyBottomRight = polyBottomRight.intersection(polygon)

    return quaterPolyTopLeft, quaterPolyTopRight, \
        quaterPolyBottomLeft, quaterPolyBottomRight


def area_q1(timeseries, nodata=-9999):
    """Area of the closed shape over the first quadrant.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarra

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return area_q1: Area of polygon that covers quadrant 1.
    """

    areas = area_season(timeseries, nodata)
    return truncate(areas[0].area)


def area_q2(timeseries, nodata=-9999):
    """Area_Q2 - Area of the closed shape over the second quadrant.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return area_q2: Area of polygon that covers quadrant 2.
    """

    areas = area_season(timeseries, nodata)
    return truncate(areas[1].area)


def area_q3(timeseries, nodata=-9999):
    """Area of the closed shape over the thrid quadrant.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return area_q3: Area of polygon that covers quadrant 3.
    """

    areas = area_season(timeseries, nodata)
    return truncate(areas[2].area)


def area_q4(timeseries, nodata=-9999):
    """Area of the closed shape over the fourth quadrant.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return area_q4: Area of polygon that covers quadrant 4.
    """

    areas = area_season(timeseries, nodata)
    return truncate(areas[3].area)


def ecc_metric(timeseries, nodata=-9999):
    """Return values close to 0 if the shape is a \
    circle and 1 if the shape is similar to a line.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return eccentricity: Eccentricity of time series after polar \
    transformation.
    """

    # filter time series
    ts = fixseries(timeseries, nodata)
    # create polygon
    polygon = create_polygon(ts)
    # get MRR
    rrec = polygon.minimum_rotated_rectangle
    minx, miny, maxx, maxy = rrec.bounds
    axis1 = maxx - minx
    axis2 = maxy - miny
    stats = numpy.array([axis1, axis2])
    return truncate((stats.min() / stats.max()))


def angle(timeseries, nodata=-9999):
    """Angle - The main angle of the closed shape created\
    by the polar visualization.
    If two angle are the same, the first one is presented.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return angle: The main angle of time series.
    """

    # filter time series
    ts = fixseries(timeseries, nodata)

    # get polar transformation info
    list_of_radius, list_of_angles = get_list_of_points(ts)

    return truncate(list_of_angles[numpy.argmax(list_of_radius)])


def gyration_radius(timeseries, nodata=-9999):
    """Gyration_radius - Equals the average distance between \
    each point inside the shape and the shape’s centroid.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return gyration_radius: Average distance between each point \
    inside the shape and the shape’s centroid.
    """

    # filtered time series
    ts = fixseries(timeseries, nodata)
    # create polygon
    polygon = create_polygon(ts)
    # get polygon centroids
    lonc, latc = polygon.centroid.xy
    # get polygon exterior coords
    x, y = polygon.exterior.coords.xy
    dist = []
    # compute distances to the centroid.
    for p in range(len(x)):
        px = x[p]
        py = y[p]
        dist = numpy.sqrt((px - lonc[0])**2 + (py - latc[0])**2)

    return truncate(numpy.mean(dist))


def polar_balance(timeseries, nodata=-9999):
    """Polar_balance - The standard deviation of the areas \
    per season, considering the 4 seasons.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return polar_balance:  Standard deviation of the areas per season.
    """

    # filter time series
    ts = fixseries(timeseries, nodata)
    # get area season
    a1, a2, a3, a4 = area_season(ts)
    return truncate(numpy.std([a1.area, a2.area, a3.area, a4.area]))


def area_ts(timeseries, nodata=-9999):
    """Area - Area of the closed shape.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return area_ts: Area of polygon.
    """

    # fix time series
    ts = fixseries(timeseries, nodata)
    # create polygon
    polygon = create_polygon(ts)

    return truncate(polygon.area)


def csi(timeseries, nodata=-9999):
    """Cell Shape Index - This is a dimensionless quantitative measure of \
    morphology, that characterize the standard deviation of an object \
    from a circle.

    :param timeseries: Time series.
    :type timeseries: numpy.ndarray

    :param nodata: nodata of the time series. Default is -9999.
    :type nodata: int

    :return shape_index: Quantitative measure of morphology.

    .. note::
        Rational of this metric:

        After polar transformation time series usually have a round shape, \
        which can be releate do cell in some cases. \
        That's why cell shape index is available here.
    """

    # filter time series
    ts = fixseries(timeseries, nodata)
    # create polygon
    polygon = create_polygon(ts).buffer(0)
    return truncate((polygon.length ** 2)/(4 * numpy.pi * polygon.area))
