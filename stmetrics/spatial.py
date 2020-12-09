import numpy
import xarray
import rasterio
from stmetrics import metrics
from numba import njit, prange, jit
from tqdm import tnrange, tqdm_notebook


def snitc(dataset, ki, m, scale=10000, iter=10, pattern="hexagonal", output="shp"):
    """This function create spatial-temporal superpixels using a Satellite \
    Image Time Series (SITS). Version 1.1

    :param image: SITS dataset.
    :type image: Rasterio dataset object or a xarray.DataArray.

    :param k: Number or desired superpixels.
    :type k: int

    :param scale: Adjust the time series, to 0-1. Necessary to distance \
    calculation.
    :type scale: int

    :param iter: Number of iterations to be performed. Default = 10.
    :type iter: int

    :param output: Type of output to be produced. Default is shp (Shapefile).
    :type output: string

    :param pattern: Type of pattern initialization. Hexagonal (default) or\
    regular (as SLIC).
    :type pattern: int

    :returns segmentation: Shapefile containing superpixels produced.
    """
    print('Simple Non-Linear Iterative Temporal Clustering V 1.3')

    if isinstance(dataset, rasterio.io.DatasetReader):
        try:
            ##READ FILE
            meta = dataset.profile #get image metadata
            transform = meta["transform"]
            crs = meta["crs"]
            img = dataset.read()
        except:
            print('Sorry we could not read your dataset.')
    elif isinstance(dataset, xarray.DataArray):
        try:
            ##READ FILE
            transform = dataset.transform
            crs = dataset.crs
            img = numpy.squeeze(dataset.values).astype(float)
        except:
            print('Sorry we could not read your dataset.')
    else:
        print("Sorry we can't read this type of file. Please use Rasterio or xarray")

    
    #Normalize data
    for band in range(img.shape[0]):
        img[numpy.isnan(img)] = 0
        img[band,:,:] = (img[band,:,:])/scale#*0.5+0.5
    
    #Get image dimensions
    bands = img.shape[0]
    rows = img.shape[1]
    columns = img.shape[2]

    if pattern == "hexagonal":
        C,S,l,d,k = init_cluster_hex(rows,columns,ki,img,bands)
    elif pattern == "regular":
        C,S,l,d,k = init_cluster_regular(rows,columns,ki,img,bands)
    else:
        print("Unknow patter. We are using hexagonal")
        C,S,l,d,k = init_cluster_hex(rows,columns,ki,img,bands)
    
    #Start clustering
    for n in range(iter):

        for kk in prange(k):
            # Get subimage around cluster
            rmin = int(numpy.floor(max(C[kk,bands]-S, 0)))
            rmax = int(numpy.floor(min(C[kk,bands]+S, rows))+1)
            cmin = int(numpy.floor(max(C[kk,bands+1]-S, 0)))
            cmax = int(numpy.floor(min(C[kk,bands+1]+S, columns))+1)

            #Create subimage 2D numpy.array
            subim = img[:,rmin:rmax,cmin:cmax]
            
            #get cluster centres
            c_series = C[kk, :subim.shape[0]]                       #Average time series
            ic = int(numpy.floor(C[kk, subim.shape[0]])) - rmin       #X-coordinate
            jc = int(numpy.floor(C[kk, subim.shape[0]+1])) - cmin     #Y-coordinate 
            
            #Calculate Spatio-temporal distance
            try:
                D = distance_fast(c_series,ic,jc,subim,S,m,rmin,cmin)
            except:
                print('dtaidistance package is not properly installed.')
                D = distance(C[kk, :], subim, S, m, rmin, cmin) #DTW regular

            subd = d[rmin:rmax,cmin:cmax]
            subl = l[rmin:rmax,cmin:cmax]
            
            #Check if Distance from new cluster is smaller than previous
            subl = numpy.where( D < subd, kk, subl)  
            subd = numpy.where( D < subd, D, subd)       
            
            #Replace the pixels that had smaller difference
            d[rmin:rmax,cmin:cmax] = subd
            l[rmin:rmax,cmin:cmax] = subl
            
        C = update_cluster(img,l,rows,columns,bands,k)     #Update Clusters

    labelled = postprocessing(l, S)     #Remove noise from segmentation
    
    if output == "shp":
        segmentation = write_pandas(labelled, transform, crs)
        return segmentation
    else:
        #Return labeled numpy.array for visualization on python
        return labelled

def distance_fast(c_series,ic,jc,subim,S,m,rmin,cmin):
    """This function computes the spatial-temporal distance between \
    two pixels using the dtw distance with C implementation.

    :param c_series: average time series of cluster
    :type c_series: numpy.ndarray

    :param ic: X coordinate of cluster center
    :type ic: int

    :param jc: Y coordinate of cluster center
    :type jc: int

    :param subim: Block of image from the cluster under analysis .
    :type subim: int

    :param S: Pattern spacing value.
    :type S: int

    :param m: Compactness value
    :type m: float

    :param rmin: Minimum row.
    :type rmin: int

    :param cmin: Minimum column.
    :type cmin: int

    :returns D: ND-Array distance
    """
    from dtaidistance import dtw
    
    #Normalizing factor
    m = m/10
    
    #Initialize submatrix
    ds = numpy.zeros([subim.shape[1],subim.shape[2]])
    
    # Tranpose matrix to allow dtw fast computation with dtaidistance
    linear = subim.transpose(1,2,0).reshape(subim.shape[1]*subim.shape[2],subim.shape[0])
    merge  = numpy.vstack((linear,c_series))

    # Compute dtw distances
    c = dtw.distance_matrix_fast(merge, block=((0, merge.shape[0]),
                                 (merge.shape[0] - 1, merge.shape[0])),
                                 compact=True, parallel=True)
    c1 = numpy.frombuffer(c)
    dc = c1.reshape(subim.shape[1], subim.shape[2])

    x = numpy.arange(subim.shape[1])
    y = numpy.arange(subim.shape[2])
    xx, yy = numpy.meshgrid(x, y, sparse=True, indexing='ij')
    ds = (((xx-ic)**2 + (yy-jc)**2)**0.5)    # Calculate Spatial Distance
    D =  (dc)/m + (ds/S)     #Calculate SPatial-temporal distance

             
    return D

def distance(c_series,ic,jc,subim,S,m,rmin,cmin):
    """This function computes the spatial-temporal distance between \
    two pixels using the DTW distance.

    :param c_series: average time series of cluster
    :type c_series: numpy.ndarray

    :param ic: X coordinate of cluster center
    :type ic: int

    :param jc: Y coordinate of cluster center
    :type jc: int

    :param subim: Block of image from the cluster under analysis .
    :type subim: int

    :param S: Pattern spacing value.
    :type S: int

    :param m: Compactness value
    :type m: float

    :param rmin: Minimum row.
    :type rmin: int

    :param cmin: Minimum column.
    :type cmin: int

    :returns D: ND-Array distance
    """
    from dtaidistance import dtw
    
    #Normalizing factor
    m = m/10

    #Initialize submatrix
    dc = numpy.zeros([subim.shape[1],subim.shape[2]])
    ds = numpy.zeros([subim.shape[1],subim.shape[2]])
            
    # Critical Loop - need parallel implementation
    for u in range(subim.shape[1]):
        for v in range(subim.shape[2]):
            a1 = subim[:,u,v]                                              # Get pixel time series 
            dc[u,v] = dtw.distance(a1.astype(float),c_series.astype(float))      # Compute DTW distance
    
    x = numpy.arange(subim.shape[1])
    y = numpy.arange(subim.shape[2])
    xx, yy = numpy.meshgrid(x, y, sparse=True, indexing='ij')
    ds = (((xx-ic)**2 + (yy-jc)**2)**0.5)                       # Calculate Spatial Distance
    
    D =  (dc)/m + (ds/S)   #Calculate SPatial-temporal distance
          
    return D

@njit(parallel = True,fastmath=True)
def update_cluster(img,la,rows,columns,bands,k):
    """This function update clusters.

    :param img: Input image.
    :type img: numpy.ndarray

    :param la: Matrix label.
    :type la: numpy.ndarray

    :param rows: Number of rows of image.
    :type rows: int

    :param columns: Number of columns of image.
    :type columns: int

    :param bands: Number of bands (lenght of time series)
    :type bands: int

    :param k: Number of superpixel.
    :type k: int

    :returns C_new: ND-array containing updated cluster centres information.
    """
    c_shape = (k,bands+3)
    
    #Allocate array info for centres
    C_new = numpy.zeros(c_shape) 
    
    #Update cluster centres with mean values
    for r in prange(rows):
        for c in range(columns):
            tmp = numpy.append(img[:,r,c],numpy.array([r,c,1]))
            kk = int(la[r,c])
            C_new[kk,:] = C_new[kk,:] + tmp
    
    #Compute mean
    for kk in prange(k):
        C_new[kk,:] = C_new[kk,:]/C_new[kk,bands+2]
    
    tmp = None
    
    return C_new


def postprocessing(raster,S):
    """Post processing function to force conectivity.

    :param raster: Labelled image
    :type raster: numpy.ndarray

    :param S: Spacing between superpixels
    :type S: int

    :param crs: Coordinate Reference Systems
    :type crs: PROJ4 dict

    :returns final: Labelled image with connectivity enforced.
    """
    import cc3d
    import fastremap
    from rasterio import features
    
    for i in range(10):
        
        raster, remapping = fastremap.renumber(raster, in_place=True)

        #Remove spourious regions generated during segmentation
        cc = cc3d.connected_components(raster.astype(dtype=numpy.uint16),
                                        connectivity=6,
                                        out_dtype=numpy.uint32)

        T = int((S**2)/2) 

        #Use Connectivity as 4 to avoid undesired connections     
        raster = features.sieve(cc.astype(dtype=rasterio.int32),T,
                                out=numpy.zeros(cc.shape,
                                                dtype = rasterio.int32),
                                connectivity = 4)
    
    return raster

def write_pandas(segmentation, transform, crs):
    """This function creates the shapefile of the segmentation produced.

    :param segmentation: Segmentation array
    :type segmentation: numpy.ndarray

    :param transform: Transformation parameters.
    :type transform: list

    :param crs: Coordinate Reference Systems
    :type crs: PROJ4 dict

    :returns gdf: Segmentation as a geopandas geodataframe.
    """
    import numpy
    import geopandas
    import rasterio.features
    from shapely.geometry import shape   
    import geopandas

    mypoly=[]

    #Loop to oconvert raster conneted components to 
    #polygons using rasterio features
    for vec in rasterio.features.shapes(segmentation.astype(dtype = numpy.float32),
                                        transform = transform):
        mypoly.append(shape(vec[0]))
        
    gdf = geopandas.GeoDataFrame(geometry=mypoly,crs=crs)
    gdf.crs = crs

    mypoly = None

    return gdf

@njit(fastmath=True)
def init_cluster_hex(rows,columns,ki,img,bands):
    """This function initialize the clusters using a hexagonal pattern.

    :param rows: Number of rows of image.
    :type rows: int

    :param columns: Number of columns of image.
    :type columns: int

    :param ki: Number of desired superpixel.
    :type ki: int

    :param img: Input image.
    :type img: numpy.ndarray

    :param bands: Number of bands (lenght of time series)
    :type bands: int

    :returns C: ND-array containing cluster centres information.

    :returns S: Spacing between clusters.

    :returns l: Matrix label.

    :returns d: Distance matrix from cluster centres.

    :returns k: Number of superpixels that will be produced.
    """
    N = rows * columns
    
    #Setting up SNITC
    S = (rows*columns / (ki * (3**0.5)/2))**0.5
    
    #Get nodes per row allowing a half column margin at one end that alternates
    nodeColumns = round(columns/S - 0.5)

    #Given an integer number of nodes per row recompute S
    S = columns/(nodeColumns + 0.5)

    # Get number of rows of nodes allowing 0.5 row margin top and bottom
    nodeRows = round(rows/((3)**0.5/2*S))
    vSpacing = rows/nodeRows

    # Recompute k
    k = nodeRows * nodeColumns
    c_shape = (k,bands+3)
    # Allocate memory and initialise clusters, labels and distances.
    C = numpy.zeros(c_shape)                 # Cluster centre data  1:times is mean on each band of series
                                                 # times+1 and times+2 is row, col of centre, times+3 is No of pixels
    l = -numpy.ones(img[0,:,:].shape)              # Matrix labels.
    d = numpy.full(img[0,:,:].shape, numpy.inf)    # Pixel distance matrix from cluster centres.

    # Initialise grid
    kk = 0;
    r = vSpacing/2;
    for ri in prange(nodeRows):
        x = ri
        if x % 2:
            c = S/2
        else:
            c = S

        for ci in range(nodeColumns):
            cc = int(numpy.floor(c)); rr = int(numpy.floor(r))
            ts = img[:,rr,cc]
            st = numpy.append(ts,[rr,cc,0])
            C[kk, :] = st
            c = c+S
            kk = kk+1

        r = r+vSpacing
    
    st = None
    #Cast S
    S = round(S)
    
    return C,S,l,d,k

@njit(fastmath=True)
def init_cluster_regular(rows,columns,ki,img,bands):
    """This function initialize the clusters using a square pattern.

    :param rows: Number of rows of image.
    :type rows: int

    :param columns: Number of columns of image.
    :type columns: int

    :param ki: Number of desired superpixel.
    :type ki: int

    :param img: Input image.
    :type img: numpy.ndarray

    :param bands: Number of bands (lenght of time series)
    :type bands: int

    :returns C: ND-array containing cluster centres information.

    :returns S: Spacing between clusters.

    :returns l: Matrix label.

    :returns d: Distance matrix from cluster centres.

    :returns k: Number of superpixels that will be produced.
    """
    N = rows * columns
    
    #Setting up SLIC    
    S = int((N/ki)**0.5)    
    base = int(S/2)
    
    # Recompute k
    k = numpy.floor(rows/base)*numpy.floor(columns/base)

    c_shape = (k,bands+3)
    # Allocate memory and initialise clusters, labels and distances.
    # Cluster centre data  1:times is mean on each band of series
    # times+1 and times+2 is row, col of centre, times+3 is No of pixels
    C = numpy.zeros(c_shape)     

    # Matrix labels.
    l = -numpy.ones(img[0,:,:].shape)
    # Pixel distance matrix from cluster centres.           
    d = numpy.full(img[0,:,:].shape, numpy.inf)

    vSpacing = int(numpy.floor(rows / ki**0.5))
    hSpacing = int(numpy.floor(columns / ki**0.5))

    kk=0

    # Initialise grid
    for x in range(base, rows, vSpacing):
        for y in range(base, columns, hSpacing):
            cc = int(numpy.floor(y)); rr = int(numpy.floor(x))
            ts = img[:,int(x),int(y)]
            st = numpy.append(ts,[int(x),int(y),0])
            C[kk, :] = st
            kk = kk+1
            
        w = S/2
    
    st = None

    return C,S,l,d,kk


def seg_metrics(dataframe, bands, features=['mean']):
    """This function compute time metrics from a geopandas \
    with time features.
    Currently, basic, polar and fractal metrics are extracted.

    :param dataframe: Images or path were images that compose time series are.
    :type dataframe: geopandas.GeodAtaframe

    :param segmentation: feature that will be used to compute the metrics.
    :type segmentation: list

    :returns out_dataframe: Geopandas dataframe with the features added.
    :rtype out_dataframe: geopandas.Dataframe
    """
    import pandas
    from . import utils
    
    out_dataframe = dataframe.copy()

    for band in bands:
        
        df = dataframe.filter(regex=band)

        for f in features:
            
            series = df.filter(regex=f)
            
            metricas = _seg_ex_metrics(series.to_numpy().astype(float))

            header = utils.list_metrics()

            names = [i + '_' + j + '_' + k \
                        for i, j, k in zip([band] * len(header),
                                            [f] * len(header),
                                            header)]
            
            metricsdf = pandas.DataFrame(metricas, columns=names)

        out_dataframe = pandas.concat([ out_dataframe, metricsdf], axis=1)

    return out_dataframe


def _seg_ex_metrics(series):
    """This function performs the computation of the metrics using \
    multiprocessing.

    :param series: Array of time series. (Series  x Time)
    :type series: numpy.array

    :returns image:  Numpy matrix of metrics and/or image.
    """
    import multiprocessing as mp
    from . import metrics
    
    # Initialize pool
    pool = mp.Pool(mp.cpu_count())

    # use pool to compute metrics for each pixel
    # return a list of arrays
    metricas = pool.map(metrics._getmetrics,[serie for serie in series])

    # close pool
    pool.close()

    # Conver list to numpy array
    X_m = numpy.vstack(metricas)

    return X_m


def extract_features(dataset, segmentation,
                     features=['mean', 'std', 'min', 'max',
                               'area', 'perimeter', 'width',
                               'length', 'ratio', 'symmetry',
                               'compactness', 'rectangular_fit'],
                     nodata=-9999):
    """This function extracts features using polygons.
    Mean, Standard Deviation, Minimum, Maximum, Area, Perimeter, \
    Lenght/With ratio, Symmetry and Compactness are extracted \
    for each polygon.

    :param dataset: Images or path were images that compose time series are.
    :type dataset: Rasterio, Xarray.Dataset or string

    :param segmentation: Spatio-temporal Segmentation.
    :type segmentation: geopandas.Dataframe

    :param features: List of features to be extracted
    :type features: list

    :param nodata: Nodata value
    :type nodata: int

    :returns segmentation: Geopandas dataframe with the features.
    :rtype segmentation: geopandas.Dataframe
    """
    import os
    import pandas
    import rasterstats
    import xarray

    # Performing buffer to solve possible invalid polygons
    segmentation['geometry'] = segmentation['geometry'].buffer(0)

    if 'area' in features:
        segmentation["area"] = segmentation['geometry'].area
        features.remove('area')

    if 'perimeter' in features:
        segmentation["perimeter"] = segmentation['geometry'].length
        features.remove('perimeter')

    if 'ratio' in features:
        segmentation["ratio"] = segmentation['geometry'].apply(lambda g: aspect_ratio(g))
        features.remove('ratio')

    if 'symmetry' in features:
        segmentation["symmetry"] = segmentation['geometry'].apply(lambda g: symmetry(g))
        features.remove('symmetry')

    if 'compactness' in features:
        segmentation["compactness"] = segmentation['geometry'].apply(lambda g: reock_compactness(g))
        features.remove('compactness')

    if 'rectangular_fit' in features:
        segmentation["rectangular_fit"] = segmentation['geometry'].apply(lambda g: rectangular_fit(g))
        features.remove('rectangular_fit')

    if 'width' in features:
        segmentation["width"] = segmentation['geometry'].apply(lambda g: width(g))
        features.remove('width')

    if 'length' in features:
        segmentation["length"] = segmentation['geometry'].apply(lambda g: length(g))
        features.remove('length')

    if isinstance(dataset, rasterio.io.DatasetReader):

        segmentation = _exRasterio(dataset, segmentation, features, nodata)

    elif isinstance(dataset, xarray.Dataset):

        segmentation = _extract_xray(dataset,
                                     segmentation,
                                     features,
                                     nodata)

    elif os.path.exists(os.path.dirname(dataset)):
        try:
            segmentation = _extract_from_path(dataset,
                                              segmentation,
                                              features,
                                              nodata)
        except:
            print('Something went wrong!')
            return None
    else:
        print('Error! We could not extract espectral information! \
              Dataset invalid')
        return None

    return segmentation


def _exRasterio(dataset, segmentation, features, nodata):
    """This function is used to extract features from images \
    that are stored in a rasterio object.
    """
    import os
    import pandas
    import rasterstats

    geoms = segmentation.geometry.tolist()

    for i in range(dataset.count):

        band = '_'+str(i+1)

        stats = fx2parallel(dataset.read(i+1),
                            geoms, features,
                            dataset.transform,
                            int(dataset.nodata))

        names = [i + j for i, j in zip(stats.columns,
                                       [band] * len(features))]

        stats.columns = names
        segmentation = pandas.concat([segmentation,
                                      stats.reindex(segmentation.index)],
                                     axis=1)

    return segmentation


def _extract_xray(dataset, segmentation, features, nodata):
    """This function is used to extract features from images \
    that are stored in a xarray.
    """
    import pandas
    import rasterstats
    from affine import Affine

    band_list = list(dataset.data_vars)
    dates = dataset.time.values
    geoms = segmentation.geometry.tolist()

    # Fix affine transformation
    # Function from_gdal swap positions we need to fix this in a brute \
    # force approach.
    c = list(dataset[band_list[0]].transform)
    affine = Affine.from_gdal(*(c[2], c[0], c[1], c[5], c[3], c[4]))

    for key in band_list:

        attr = numpy.squeeze(dataset[key].values)

        for i in range(attr.shape[0]):
            stats = fx2parallel(attr[i, :, :], geoms, features, affine,
                                int(dataset[key].nodatavals[0]))

            names = [y + j + g + f + k for y, j, g, f, k in
                     zip([key] * len(features),
                         ['_'] * len(features),
                         [str(dates[i])] * len(features),
                         ['_'] * len(features), stats.columns)]

            stats.columns = names
            segmentation = pandas.concat([segmentation, stats], axis=1)
    
    c = None
    names = None
    
    return segmentation


def _extract_from_path(path, segmentation, features, nodata):
    """This function is used to extract features from images that\
     are stored in a folder.
    """

    import os
    import re
    import glob
    import pandas
    import rasterio
    import rasterstats

    # Read images and sort
    f_path = glob.glob(path+"*.tif")
    f_path.sort()
    geoms = segmentation.geometry.tolist()

    for f in f_path:
        dataset = rasterio.open(f)
        affine = dataset.transform

        # find datetime and att
        key = os.path.basename(f).split('.')[0]
        
        stats = fx2parallel(dataset.read(1),
                            geoms, features,
                            dataset.transform,
                            int(dataset.nodata))

        stats.columns = [y + j + g + k for y, j, g, k in
                         zip([key] * len(features),
                             ['_'] * len(features),
                             ['_'] * len(features),
                             stats.columns)]

        segmentation = pandas.concat([segmentation,
                                      stats.reindex(segmentation.index)
                                      ], axis=1)

    return segmentation


def _chunks(data, n):
    """Yield successive n-sized chunks from a slice-able iterable."""
    for i in range(0, len(data), n):
        yield data[i:i+n]


def _zonal_stats_wrapper(raster, stats, affine, nodata):
    """Wrapper for zonal stats, takes a list of features"""
    from rasterstats import zonal_stats
    import functools

    return functools.partial(zonal_stats, raster=raster, stats=stats,
                             affine=affine, nodata=nodata)


def fx2parallel(dataset, geoms, features, transform, nodata):
    """This functions allow the extraction of features."""
    import pandas
    import itertools
    import multiprocessing

    cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cores)

    _zonal_stats_partial = _zonal_stats_wrapper(dataset, features,
                                                affine=transform,
                                                nodata=nodata)

    stats_lst = p.map(_zonal_stats_partial, _chunks(geoms, (cores)))

    stats = pandas.DataFrame(list(itertools.chain(*stats_lst)))

    p.close()

    return stats


def aspect_ratio(geom):
    """This function computes the aspect ratio of a given geometry.

    The Length-Width Ratio (LW) is the ratio of the length \
    (LMBR) and the width (WMBR) of the minimum bounding \
    rectangle of a polygon.

    :param geom: Polygon geometry
    :type geom: shapely.geometry.Polygon

    :returns aspect_ratio: Polygon aspect_ratio.
    """

    from shapely.geometry import Polygon, LineString

    # get the MBR and zip coordinates into a list of point-tuples
    mbr_points = list(zip(*geom.minimum_rotated_rectangle.exterior.coords.xy))

    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length
                   for i in range(len(mbr_points) - 1)]

    # get major/minor axis measurements
    minor_axis = min(mbr_lengths)
    major_axis = max(mbr_lengths)

    return minor_axis/major_axis


def symmetry(geom):
    """
    This function computes the symmetry of a given geometry.

    Symmetry is calculated by dividing the overlapping area AO, between \
    a polygon and its reflection across the horizontal axis by the area of \
    the original polygon P. The range of this score goes between [0,1] and a \
    score closer to 1 indicates a more compact and regular geometry.

    .. math:: Symmetry = AO/A_p

    :param geom: Polygon geometry
    :type geom: shapely.geometry.Polygon

    :returns symmetry: Polygon symmetry.
    """
    from shapely import affinity

    rotated = affinity.rotate(geom, 180)

    sym_dif = geom.symmetric_difference(rotated)

    return sym_dif.area/geom.area


def reock_compactness(geom):
    """This function computes the reock compactness of a given geometry.

    The Reock Score (R) is the ratio of the area of the polygon P to the \
    area of a minimum bounding cirle (AMBC) that encloses the geometry. A \
    polygon Reock score falls within the range of [0,1] and high values \
    indicates a more compact district.

    .. math:: Reock = A_p/A_{MBC}

    :param geom: Polygon geometry
    :type geom: shapely.geometry.Polygon

    :returns reock: Polygon reock compactness.
    
    .. Tip:: To know more about it:

        Reock, Ernest C. 1961. “A note: Measuring compactness as a requirement \
        of legislative apportionment.” Midwest Journal of Political Science \
        1(5), 70–74.
    """
    import pointpats
    from shapely.geometry import Point

    points = list(zip(*geom.minimum_rotated_rectangle.exterior.coords.xy))
    (radius, center), _, _, _ = pointpats.skyum(points)
    mbc_poly = Point(*center).buffer(radius)

    return geom.area/mbc_poly.area


def rectangular_fit(geom):
    """This functions computes the rectangular_fit of a geometry. \
    Rectangular fit is defined as:

    .. math:: RectFit = (AR - AD) / AO

    where AO is the area of the original object, AR is the area of the \
    equal rectangle (AO = AR) and AD is the overlaid difference between \
    the equal rectangle and the original object (Sun et al. 2015).

    :param geom: Polygon geometry
    :type geom: shapely.geometry.Polygon

    :returns rectangular_fit: Polygon rectangular fit.

    .. Tip:: To know more about it:

        Sun, Z., Fang, H., Deng, M., Chen, A., Yue, P. and Di, L.. "Regular \
        Shape Similarity Index: A Novel Index for Accurate Extraction of Regular \
        Objects From Remote Sensing Images," IEEE Transactions on Geoscience and \
        Remote Sensing, v.53, 2015, p. 3737. doi:10.1109/TGRS.2014.2382566
    """

    mrc = geom.minimum_rotated_rectangle

    return geom.symmetric_difference(mrc).area/geom.area


def width(geom):
    """This functions computes the width of a geometry.

    :param geom: Polygon geometry
    :type geom: shapely.geometry.Polygon

    :returns width: Polygon width.
    """

    minx, miny, maxx, maxy = geom.bounds

    return maxx - minx


def length(geom):
    """This functions computes the lenght of a geometry.

    :param geom: Polygon geometry
    :type geom: shapely.geometry.Polygon

    :returns length: Polygon length.
    """

    minx, miny, maxx, maxy = geom.bounds

    return maxy - miny
