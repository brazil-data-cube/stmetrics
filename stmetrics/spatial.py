import numpy
import rasterio
from stmetrics import metrics

def snitc(dataset,ki,m):

    """
    
    This function create spatial-temporal superpixels using a Satellite Image Time Series (SITS).

    Keyword arguments:
    ------------------
        image : Rasterio dataset object
            Input image
        k : int
            Number or desired superpixels
        m : float
            Compactness factor
        factor: float
            Adjust the time series distance, usually 100 performs best.

    Returns
    -------
        Shapefile containing superpixels produced.
    
    """
    import os
    from tqdm import tnrange, tqdm_notebook

    print('Simple Non-Linear Iterative Temporal Clustering V 1.0')
    name = os.path.basename(dataset.name)[:-4]

    ##READ FILE
    img = dataset.read()
    
    meta = dataset.profile #get image metadata
    transform = meta["transform"]
    crs = meta["crs"]
    
    #Normalize data
    for band in range(img.shape[0]):
        img[numpy.isnan(img)] = 0
        img[band,:,:] = img[band,:,:]#*0.5+0.5
    
    #Get image dimensions
    bands = img.shape[0]
    rows = img.shape[1]
    columns = img.shape[2]

    C,S,l,d,k = init_cluster_hex(rows,columns,ki,img,bands)
    
    #Start clustering
    for n in tnrange(10):
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
                print('Please, check our documentation and follow the steps.')
                print('This will take too much time to finish, use this time to properly install =D')
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
    labelled = postprocessing(l,S)                 #Remove noise from segmentation
    
    segmentation = write_pandas(labelled, meta)
    
    #print('Writing raster')
    #write_raster(labelled, meta, name, ki, m)
        
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
    
    #f = subim.shape[0]
    m = m/10
    #Initialize submatrix
    ds = numpy.zeros([subim.shape[1],subim.shape[2]])

    #get cluster centres
    a2 = C[:subim.shape[0]]                                #Average time series
    ic = (int(numpy.floor(C[subim.shape[0]])) - rmin)         #X-coordinate
    jc = (int(numpy.floor(C[subim.shape[0]+1])) - cmin)       #Y-coordinate
    
    linear = subim.transpose(1,2,0).reshape(subim.shape[1]*subim.shape[2],subim.shape[0])
    merge  = numpy.vstack((linear,a2))

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
    #f = subim.shape[0]
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
    
    for smooth in range(3):
        #Remove spourious regions generated during segmentation
        cc = cc3d.connected_components(raster.astype(dtype=numpy.uint16), connectivity=6, out_dtype=numpy.uint32)

        T = int((S**2)/2) 

        #Use Connectivity as 4 to avoid undesired connections     
        raster = features.sieve(cc.astype(dtype=numpy.int32),T,connectivity = 4)
    
    return raster

def write_pandas(segmentation, meta):

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


    #Get-Set transform and CRS
    transform = meta["transform"]
    crs = meta["crs"]
    
    import geopandas

    mypoly=[]

    for vec in rasterio.features.shapes(segmentation.astype(dtype = numpy.float32), transform = transform):
        mypoly.append(shape(vec[0]))
        
    gdf = geopandas.GeoDataFrame(geometry=mypoly)

    return gdf
    
def write_raster(segmentation, meta, name, k, m):
    
    """
    
    This function creates a TUF file of the segmentation produced.

    Keyword arguments:
    ------------------
        segmentation : numpy.ndarray
            Segmentation array
        meta : int
            Metadata of the original image
        name: string
            Output name
        k:
            Number of desired superpixels
        m: 
            Compactness

    Returns
    -------
    Segmentation as TIF file.
    
    """

    #Adjust metadata to flush temporary file to 1
    meta['count'] = 1
    # change the data type to float rather than integer
    meta['dtype'] = "uint32"
    meta['nodata'] = 0
    
    with rasterio.open(name+"_"+str(k)+"_"+str(m)+".tif", 'w', **meta) as dst:
        dst.write(segmentation.astype(dtype = numpy.uint32), 1)
        
    return None

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

def extract_features(dataset,segmentation,features = ['mean','std','min','max','area','perimeter','ratio','symmetry','compactness']):
    """
    This function extracts features using polygons.
    Mean, Standard Deviation, Minimum, Maximum, Area, Perimeter, Lenght/With ratio, Symmetry and Compactness are extracted for each polygon.
    Nodata information is extracted from raster metadata.

    Keyword arguments:
    ------------------
        image : rasterio dataset
        segmentation : geopandas dataframe

    Returns
    -------
        geopandas.Dataframe:
            segmentation
    """
    import pandas
    import rasterstats

    affine = dataset.transform
    
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
        
    if any(feat in features for feat in ('mean','std','min','max')):
        for i in range(dataset.count):
            band = '_'+str(i+1)
            stats = pandas.DataFrame(rasterstats.zonal_stats(segmentation, dataset.read(i+1), affine=affine, stats=features, nodata=int(dataset.nodata)))
            names = [i + j for i, j in zip(stats.columns, [band] * len(features))]
            stats.columns = names
            segmentation = pandas.concat([segmentation, stats.reindex(segmentation.index)], axis=1)
        
    return segmentation

def seg_metrics(dataframe,feature='mean',merge=True):
    
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

    series = dataframe.filter(regex=feature)
    metricas = seg_exmetrics(series.to_numpy())
    
    header=["Mean", "Max", "Min", "Std", "Sum","Amplitude","First_slope","Area","Area_s1","Area_s2","Area_s3","Area_s4","Circle","Gyration","Polar_balance","Angle", "DFA","Hurst","Katz"]
    
    metricsdf = pandas.DataFrame(metricas,columns = header)
    
    if merge==True:
        out_dataframe = pandas.concat([dataframe, metricsdf], axis=1)
        return out_dataframe
    else:
        return metricsdf


def seg_exmetrics(series):
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
    metricas = pool.map(metrics.get_metrics,[serie for serie in series])
        
    #close pool
    pool.close()    
        
    #Conver list to numpy array
    X_m = numpy.vstack(metricas)
        
    return X_m

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
        LW : double

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