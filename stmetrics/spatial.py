import numpy
import xarray
import rasterio
from stmetrics import metrics

def snitc(dataset, ki, m, scale=10000, iter=10, pattern="hexagonal"):

    """
    
    This function create spatial-temporal superpixels using a Satellite Image Time Series (SITS).

    Keyword arguments:
    ------------------
        image : Rasterio dataset object or a xarray.DataArray
            Input image
        k : int
            Number or desired superpixels
        m : float
            Compactness factor
        scale: int
            Adjust the time series, to 0-1.
        iter: int
            Number of iterations to be performed.
        pattern: string
            Type of pattern initialization. it can be hexagonal (default) or regular (as SLIC).

    Returns
    -------
        Shapefile containing superpixels produced.
    
    """
    import os
    from tqdm import tnrange, tqdm_notebook

    print('Simple Non-Linear Iterative Temporal Clustering V 1.1')
    name = os.path.basename(dataset.name)[:-4]

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
    for n in tnrange(iter):
        residual_error = 0
        for kk in range(k):
            # Get subimage around cluster
            rmin = int(numpy.floor(max(C[kk,bands]-S, 0)))
            rmax = int(numpy.floor(min(C[kk,bands]+S, rows))+1)
            cmin = int(numpy.floor(max(C[kk,bands+1]-S, 0)))
            cmax = int(numpy.floor(min(C[kk,bands+1]+S, columns))+1)

            #Create subimage 2D numpy.array
            subim = img[:,rmin:rmax,cmin:cmax]
            
            #Calculate Spatio-temporal distance
            try:
                D = distance_fast(C[kk, :], subim, S, m, rmin, cmin) #DTW fast
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
            
        C,_ = update_cluster(C,img,l,rows,columns,bands,k,residual_error)          #Update Clusters

        
    #print('Fixing segmentation')
    labelled = postprocessing(l, S)                 #Remove noise from segmentation
    
    segmentation = write_pandas(labelled, transform, crs)
            
    return segmentation                                 #Return labeled numpy.array for visualization on python

def distance_fast(C, subim, S, m, rmin, cmin):
    """
    
    This function computes the spatial-temporal distance between
    two pixels using the dtw distance with C implementation.

    Keyword arguments:
    ------------------
        C : numpy.ndarray
            ND-array containing cluster centres information
        subim : numpy.ndarray  
            Cluster under analisis
        S : float  
            Spacing
        m : float 
            Compactness
        rmin : float
            Minimum row
        cmin : float
            Minimum column
        factor : float
            Corrective factor

    Returns
    -------
    D: numpy.ndarray
        ND-Array distance
    
    """
    from dtaidistance import dtw
    
    #Normalizing factor
    m = m/10
    
    #Initialize submatrix
    ds = numpy.zeros([subim.shape[1],subim.shape[2]])

    #get cluster centres
    a2 = C[:subim.shape[0]]                                #Average time series
    ic = (int(numpy.floor(C[subim.shape[0]])) - rmin)         #X-coordinate
    jc = (int(numpy.floor(C[subim.shape[0]+1])) - cmin)       #Y-coordinate
    
    # Tranpose matrix to allow dtw fast computation with dtaidistance
    linear = subim.transpose(1,2,0).reshape(subim.shape[1]*subim.shape[2],subim.shape[0])
    merge  = numpy.vstack((linear,a2))

    #Compute dtw distances
    c = dtw.distance_matrix_fast(merge, block=((0, merge.shape[0]), (merge.shape[0]-1,merge.shape[0])), compact=True, parallel=True)
    dc = c.reshape(subim.shape[1],subim.shape[2])
    
    # Critical Loop - need parallel implementation
    for u in range(subim.shape[1]):
        for v in range(subim.shape[2]):
            ds[u,v] = (((u-ic)**2 + (v-jc)**2)**0.5)                         #Calculate Spatial Distance
    
    D =  (dc)/m + (ds/S)                                #Calculate SPatial-temporal distance
             
    return D

def distance(C, subim, S, m, rmin, cmin):
    """
    
    This function computes the spatial-temporal distance between
    two pixels using the DTW distance.

    Keyword arguments:
    ------------------
        C : numpy.ndarray
            ND-array containing cluster centres information
        subim : numpy.ndarray  
            Cluster under analisis
        S : float  
            Spacing
        m : float 
            Compactness
        rmin : float
            Minimum row
        cmin : float
            Minimum column
        factor : float
            Corrective factor

    Returns
    -------
    D: numpy.ndarray
        ND-Array distance
    
    """
    from dtaidistance import dtw
    
    #Normalizing factor
    m = m/10

    #Initialize submatrix
    dc = numpy.zeros([subim.shape[1],subim.shape[2]])
    ds = numpy.zeros([subim.shape[1],subim.shape[2]])
            
    #get cluster centres
    a2 = C[:subim.shape[0]]                                #Average time series
    ic = (int(numpy.floor(C[subim.shape[0]])) - rmin)         #X-coordinate
    jc = (int(numpy.floor(C[subim.shape[0]+1])) - cmin)       #Y-coordinate
    
    # Critical Loop - need parallel implementation
    for u in range(subim.shape[1]):
        for v in range(subim.shape[2]):
            a1 = subim[:,u,v]                                              # Get pixel time series 
            dc[u,v] = dtw.distance(a1.astype(float),a2.astype(float))      # Compute DTW distance
            ds[u,v] = (((u-ic)**2 + (v-jc)**2)**0.5)                       # Calculate Spatial Distance
    
    D =  (dc)/m + (ds/S)   #Calculate SPatial-temporal distance
          
    return D

def update_cluster(C,img,l,rows,columns,bands,k,residual_error):

    """
    
    This function update clusters' informations.

    Keyword arguments:
    ------------------
        C : numpy.ndarray
            ND-array containing cluster centres information
        img : numpy.ndarray  
            Input image
        L : float  
            Spacing
        rows : float 
            Number of rows in the image
        columns : float
            Number of columns in the image
        band : float
            Number of bands
        k : float
            Number os superpixels
        residual_error:
            residual_error from previous iteration

    Returns
    -------
    C: numpy.ndarray
        Updated cluster centres information.
    
    """
    
    #Allocate array info for centres
    C_new = numpy.zeros([k,bands+3]).astype(float) 
    error = numpy.zeros([k,1]).astype(float)

    #Update cluster centres with mean values
    for r in range(rows):
        for c in range(columns):
            tmp = numpy.append(img[:,r,c],numpy.array([r,c,1]))
            kk = l[r,c].astype(int)
            C_new[kk,:] = C_new[kk,:] + tmp
  
    #Compute mean
    for kk in range(k):
        C_new[kk,:] = C_new[kk,:]/C_new[kk,bands+2]
        
        partial_error = C[kk,:] - C_new[kk,:]
     
        error[kk,:] = residual_error + numpy.sqrt(partial_error.dot(partial_error.transpose()))
        
    residual_error = numpy.mean(error)
        
    return C_new,residual_error


def postprocessing(raster,S):

    """
    
    This function forces conectivity.

    Keyword arguments:
    ------------------
        raster : numpy.ndarray
            Labelled image
        S : int
            Spacing
    Returns
    -------
    final: numpy.ndarray
        Segmentation result
    
    """
    import cc3d
    import fastremap
    from rasterio import features
    
    for i in range(10):
        
        raster, remapping = fastremap.renumber(raster, in_place=True)

        #Remove spourious regions generated during segmentation
        cc = cc3d.connected_components(raster.astype(dtype=numpy.uint16), connectivity=6, out_dtype=numpy.uint32)

        T = int((S**2)/2) 

        #Use Connectivity as 4 to avoid undesired connections     
        raster = features.sieve(cc.astype(dtype=rasterio.int32),T, out=numpy.zeros(cc.shape, dtype = rasterio.int32), connectivity = 4)
    
    return raster

def write_pandas(segmentation, transform, crs):

    """
    
    This function creates the shapefile of the segmentation produced.

    Keyword arguments:
    ------------------
        segmentation : numpy.ndarray
            Segmentation array
        meta : int
            Metadata of the original image
    Returns
    -------
    Segmentation geopandas
    
    """
    import numpy
    import geopandas
    import rasterio.features
    from shapely.geometry import shape   
    import geopandas

    mypoly=[]

    #Loop to oconvert raster conneted components to polygons using rasterio features
    for vec in rasterio.features.shapes(segmentation.astype(dtype = numpy.float32), transform = transform):
        mypoly.append(shape(vec[0]))
        
    gdf = geopandas.GeoDataFrame(geometry=mypoly,crs=crs)
    gdf.crs = crs
    return gdf

def init_cluster_hex(rows,columns,ki,img,bands):

    """
    
    This function initialize the clusters using a hexagonal pattern.

    Keyword arguments:
    ------------------
        img : numpy.ndarray
            Input image
        bands : int
            Number of bands (lenght of time series)
        rows: int
            Number of rows
        columns: int
            Number of columns
        ki:
            Number of desired superpixel

    Returns
    -------
        C : numpy.ndarray
            ND-array containing cluster centres information
        S : float  
            Spacing
        l : numpy.ndarray 
            Matrix label
        d : numpy.ndarray 
            Distance matrix from cluster centres
        k : int
            Number of superpixels that will be produced
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

    # Allocate memory and initialise clusters, labels and distances.
    C = numpy.zeros([k,bands+3])                 # Cluster centre data  1:times is mean on each band of series
                                                 # times+1 and times+2 is row, col of centre, times+3 is No of pixels
    l = -numpy.ones([rows,columns])              # Matrix labels.
    d = numpy.full([rows,columns], numpy.inf)    # Pixel distance matrix from cluster centres.

    # Initialise grid
    kk = 0;
    r = vSpacing/2;
    for ri in range(nodeRows):
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
    
    #Cast S
    S = round(S)
    
    return C,S,l,d,k

def init_cluster_regular(rows,columns,ki,img,bands):

    """
    
    This function initialize the clusters using a square pattern.

    Keyword arguments:
    ------------------
        img : numpy.ndarray
            Input image
        bands : int
            Number of bands (lenght of time series)
        rows: int
            Number of rows
        columns: int
            Number of columns
        ki:
            Number of desired superpixel

    Returns
    -------
        C : numpy.ndarray
            ND-array containing cluster centres information
        S : float  
            Spacing
        l : numpy.ndarray 
            Matrix label
        d : numpy.ndarray 
            Distance matrix from cluster centres
        k : int
            Number of superpixels that will be produced
    """

    N = rows * columns
    
    #Setting up SLIC    
    S = int((N/ki)**0.5)    
    base = int(S/2)
    
    # Recompute k
    k = numpy.floor(rows/base)*numpy.floor(columns/base)

    # Allocate memory and initialise clusters, labels and distances.
    C = numpy.zeros([int(k),bands+3])            # Cluster centre data  1:times is mean on each band of series
                                              # times+1 and times+2 is row, col of centre, times+3 is No of pixels
    l = -numpy.ones([rows,columns])              # Matrix labels.
    d = numpy.full([rows,columns], numpy.inf)       # Pixel distance matrix from cluster centres.

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
        
    return C,int(S),l,d,int(kk)

def seg_metrics(dataframe,feature=['mean'],merge=True):
    
    """
    This function compute time metrics from a geopandas with time features.
    Basic, polar and fractal metrics.
    
    Keyword arguments:
    ------------------
        dataframe : geodataframe
        feature : feature that will be used to compute the metrics. Usually mean.

    Returns
    -------
        geopandas.Dataframe
    
    """
    import pandas

    for f in feature:
        series = dataframe.filter(regex=f)
        metricas = _seg_ex_metrics(series.to_numpy())

        header=['max_ts','min_ts','mean_ts','std_ts','sum_ts','amplitude_ts','mse_ts','fslope_ts','skew_ts','amd_ts','abs_sum_ts','iqr_ts','fqr_ts','tqr_ts','sqr_ts','ecc_metric','gyration_radius','area_ts','polar_balance','angle','area_q1','area_q2','area_q3','area_q4','dfa_fd','hurst_exp','katz_fd']
        
        metricsdf = pandas.DataFrame(metricas,columns = header)
    
    if merge==True:
        out_dataframe = pandas.concat([dataframe, metricsdf], axis=1)
        return out_dataframe
    else:
        return metricsdf


def _seg_ex_metrics(series):
    """
    This function performs the computation of the metrics using multiprocessing.

    Keyword arguments:
    ------------------

    image : numpy.array
        Array of time series. (Series  x Time)
    merge : Boolean
        Indicate if the matrix of features should be merged with the input matrix.
    
    Returns
    -------
        image : numpy.array
            Numpy matrix of metrics and/or image.

    """
    import multiprocessing as mp

    #Initialize pool
    pool = mp.Pool(mp.cpu_count())
        
    #use pool to compute metrics for each pixel
    #return a list of arrays
    metricas = pool.map(metrics._sitsmetrics,[serie for serie in series])
        
    #close pool
    pool.close()    
        
    #Conver list to numpy array
    X_m = numpy.vstack(metricas)
        
    return X_m

def extract_features(dataset,segmentation,features = ['mean','std','min','max','area','perimeter','width','length','ratio','symmetry','compactness','rectangular_fit'], nodata = -9999):
    """
    This function extracts features using polygons.
    Mean, Standard Deviation, Minimum, Maximum, Area, Perimeter, Lenght/With ratio, Symmetry and Compactness are extracted for each polygon.
    Nodata value can be is extracted from raster metadata.

    Keyword arguments:
    ------------------
        image : rasterio dataset
        segmentation : geopandas dataframe

    Returns
    -------
        geopandas.Dataframe:
            segmentation
    """
    import os
    import pandas
    import rasterstats
    import xarray

    #Performing buffer to solve possible invalid polygons
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

        segmentation = _exRasterio(dataset,segmentation, features, nodata)

    elif isinstance(dataset, xarray.Dataset): 

        segmentation = _extract_xray(dataset, segmentation, features, nodata)

    elif os.path.exists(os.path.dirname(dataset)):
        try:
            segmentation = _extract_from_path(dataset, segmentation, features, nodata)
        except:
            print('Something went wrong!')
            return None
    else:
        print('Error! We could not extract espectral information! Dataset invalid')
        return None
    
    return segmentation


def _exRasterio(dataset,segmentation, features, nodata):
    """
    This function is used to extract features from images that are stored in a rasterio object.
    """

    import os
    import pandas
    import rasterstats

    geoms = segmentation.geometry.tolist()

    for i in range(dataset.count):
        band = '_'+str(i+1)
        stats = fx2parallel(dataset.read(i+1), geoms, features, dataset.transform, int(dataset.nodata))
        #stats = pandas.DataFrame(rasterstats.zonal_stats(segmentation, dataset.read(i+1), affin=edataset.transform, stats=features, nodata=int(dataset.nodata)))
        names = [i + j for i, j in zip(stats.columns, [band] * len(features))]
        stats.columns = names
        segmentation = pandas.concat([segmentation, stats.reindex(segmentation.index)], axis=1)

    return segmentation

def _extract_xray(dataset, segmentation, features, nodata):
    """
    This function is used to extract features from images that are stored in a xarray.
    """
    
    import numpy
    import pandas
    import rasterstats
    from affine import Affine
    
    band_list = list(dataset.data_vars)
    dates = dataset.time.values
    geoms = segmentation.geometry.tolist()

    #Fix affine transformation
    #Function from_gdal swap positions we need to fix this in a brute force approach.
    c = list(dataset[band_list[0]].transform)
    affine = Affine.from_gdal(*(c[2],c[0],c[1], c[5], c[3], c[4]))

    for key in band_list:
        attr = numpy.squeeze(dataset[key].values)
        for i in range(attr.shape[0]):
            stats = fx2parallel(attr[i,:,:], geoms, features, affine, int(dataset[key].nodatavals[0]))
            #stats = pandas.DataFrame(rasterstats.zonal_stats(segmentation, attr[i,:,:], stats = features, affine = affine, nodata=-99999))
            names = [y + j + g + f+ k for y, j, g, f, k in zip([key] * len(features), ['_'] * len(features), [str(dates[i])] * len(features), ['_'] * len(features), stats.columns)]
            stats.columns = names
            segmentation = pandas.concat([segmentation, stats], axis=1)
            
    return segmentation

def _extract_from_path(path,segmentation,features,nodata):
    """
    This function is used to extract features from images that are stored in a folder.
    """
    
    import os
    import re
    import glob
    import pandas
    import rasterio
    import rasterstats
        
    #Read images and sort
    f_path = glob.glob(path+"*.tif")   
    f_path.sort()
    geoms = segmentation.geometry.tolist()

    for f in f_path:
        
        dataset = rasterio.open(f)
        affine = dataset.transform

        #find datetime and att
        key = os.path.basename(f).split('_')[-1][:-4]
        match = re.findall(r'\d{4}-\d{2}-\d{2}', f)[-1]
        stats = fx2parallel(dataset.read(1), geoms, features,dataset.transform, int(dataset.nodata))

        #stat = pandas.DataFrame(rasterstats.zonal_stats(segmentation, dataset.read(1), stats = features, affine = affine, nodata=dataset.nodata, all_touched=False))
        stats.columns = [y + j + g + f + k for y, j, g, f, k in zip([key] * len(features), ['_'] * len(features), [match] * len(features), ['_'] * len(features), stats.columns)]
        segmentation = pandas.concat([segmentation, stats.reindex(segmentation.index)], axis=1)

    return segmentation

def _chunks(data, n):
    """Yield successive n-sized chunks from a slice-able iterable."""
    for i in range(0, len(data), n):
        yield data[i:i+n]

def _zonal_stats_wrapper(raster, stats, affine, nodata):
    """Wrapper for zonal stats, takes a list of features"""
    from rasterstats import zonal_stats
    import functools
    return functools.partial(zonal_stats,raster=raster, stats=stats, affine = affine, nodata=nodata)

def fx2parallel(dataset, geoms, features, transform, nodata):
    """
    This functions allow the extraction of features.
    """
    import pandas
    import itertools
    import multiprocessing

    cores = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cores - 1)

    _zonal_stats_partial = _zonal_stats_wrapper(dataset, features, affine=transform, nodata=nodata)  
    stats_lst = p.map(_zonal_stats_partial, _chunks(geoms, (cores - 1)))
    stats = pandas.DataFrame(list(itertools.chain(*stats_lst)))
    
    p.close()    
    
    return stats

def aspect_ratio(geom):
    """
    This function computes the aspect ratio of a given geometry.
    
    The Length-Width Ratio (LW) is the ratio of the length (LMBR) and the width (WMBR) of the minimum bounding rectangle of a polygon. 
    
    Keyword arguments:
    ------------------
    geom: shapely.geometry.Polygon
        Polygon geometry
    Returns
    -------
        aspect_ratio : double
    """

    from shapely.geometry import Polygon, LineString

    # get the minimum bounding rectangle and zip coordinates into a list of point-tuples
    mbr_points = list(zip(*geom.minimum_rotated_rectangle.exterior.coords.xy))

    # calculate the length of each side of the minimum bounding rectangle
    mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]

    # get major/minor axis measurements
    minor_axis = min(mbr_lengths)
    major_axis = max(mbr_lengths)
    
    return minor_axis/major_axis

def symmetry(geom):
    """
    This function computes the symmetry of a given geometry.

    Symmetry is calculated by dividing the overlapping area AO, between a polygon and its reflection across the horizontal axis by the area of the original polygon P.
    The range of this score goes between [0,1] and a score closer to 1 indicates a more compact and regular geometry.

    .. math:: Symmetry = AO/A_p
    
    Keyword arguments:
    ------------------
    
    geom: shapely.geometry.Polygon
       Polygon geometry
    Returns
    -------
        symmetry : double
    """
    from shapely import affinity

    rotated = affinity.rotate(geom, 180)
    
    sym_dif = geom.symmetric_difference(rotated)
    
    return sym_dif.area/geom.area

def reock_compactness(geom):
    """
    This function computes the reock compactness of a given geometry.

    The Reock Score (R) is the ratio of the area of the polygon P to the area of a minimum bounding cirle (AMBC) that encloses the geometry. 
    A polygon Reock score falls within the range of [0,1] and high values indicates a more compact district.

    .. math:: Reock = A_p/A_{MBC}
    
    Keyword arguments:
    ------------------

    geom: shapely.geometry.Polygon
        Polygon geometry
    Returns
       reock compactness : double

    Reock, Ernest C. 1961. “A note: Measuring compactness as a requirement of legislative apportionment.” Midwest Journal of Political Science 1(5), 70–74.
    """
    import pointpats
    from shapely.geometry import Point
    
    points = list(zip(*geom.minimum_rotated_rectangle.exterior.coords.xy))
    (radius, center), _, _, _ = pointpats.skyum(points)
    mbc_poly = Point(*center).buffer(radius)
    
    return geom.area/mbc_poly.area

def rectangular_fit(geom):
    '''
    This functions computes the rectangular_fit of a geometry. Rectangular fit is defined as:

    .. math:: RectFit = (AR - AD) / AO

    where AO is the area of the original object, AR is the area of the equal rectangle (AO = AR) and AD is the overlaid difference between the equal rectangle and the original object (Sun et al. 2015).
    

    Keyword arguments:
    ------------------

    geom: shapely.geometry.Polygon
        Polygon geometry
    Returns
        rectangular_fit : double

    Sun, Z., Fang, H., Deng, M., Chen, A., Yue, P. and Di, L.. "Regular Shape Similarity Index: A Novel Index for Accurate Extraction of Regular Objects From Remote Sensing Images," IEEE Transactions on Geoscience and Remote Sensing, v.53, 2015, p. 3737. doi:10.1109/TGRS.2014.2382566
    '''  
    
    mrc = geom.minimum_rotated_rectangle
    
    return geom.symmetric_difference(mrc).area/geom.area

def width(geom):
    '''
    This functions computes the width of a geometry.

    Keyword arguments:
    ------------------

    geom: shapely.geometry.Polygon
        Polygon geometry
    Returns
        width : double
    '''
    
    minx, miny, maxx, maxy = geom.bounds

    return maxx - minx

def length(geom):
    '''
    This functions computes the lenght of a geometry.

    Keyword arguments:
    ------------------

    geom: shapely.geometry.Polygon
        Polygon geometry
    Returns
        length : double
    '''
    
    minx, miny, maxx, maxy = geom.bounds

    return maxy - miny