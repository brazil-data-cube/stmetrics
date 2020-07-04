import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.signal import savgol_filter,medfilt,find_peaks


def get_filtered_series(timeseries,window,treshold):
    
    """
    
    This function filter the input timeseries using the Savitzky-Golay method, using \\
    a similar approach to the Timesat. 
    
    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        treshold : float
            Minimum growing that will be used to detect a new cycle.
        window: integer (odd number) default 7
            Size of the window used for filtering with the Savitzky-Golay 
    Returns
    -------
    numpy.ndarray:
        filtered timeseries (sg)
    
    """
    
    min_win = int(numpy.floor(window/2))
    y = numpy.empty([min_win, timeseries.shape[0]])
    
    pos = 0
    for i in range(window,3,-2):
        y[pos,:] = savgol_filter(timeseries,i,2)
        pos+=1

    diff = abs(y[0]/timeseries)
    diff[numpy.isnan(diff)]=0
    diff[numpy.isinf(diff)]=0
    
    max_diff = abs(numpy.max(diff))

    level = 0
    sg = y[0]

    while max_diff > treshold and level < pos-1:
        new = y[level+1,:]
        sg = numpy.where(diff>treshold,new,sg)
        diff = abs(sg/timeseries)
        diff[numpy.isnan(diff)]=0
        diff[numpy.isinf(diff)]=0
        max_diff = abs(numpy.max(diff))
        level += 1
    
    return sg

def get_ts_metrics(y,a,c,l_mi,b,d,r_mi,tresh):

    """
    
    This function compute the phenological metrics
    
    Reference: 

    Keyword arguments:
       
    Returns
    -------
    numpy.ndarray:
        array of peaks
    """

    #Interpolate a, b, c and d positions
    #Fine adjustment
    xp = numpy.arange(0,len(y))
    start_val = numpy.interp(a, xp, y)
    end_val = numpy.interp(b, xp, y)
    yc = numpy.interp(c, xp, y)
    yd = numpy.interp(d, xp, y)
    
    #Compute other metrics
    l_derivada = abs(start_val-yc)/abs(a-c)
    r_derivada = abs(end_val-yd)/abs(b-d)  
    lenght = abs(a-b)
    Base_val = (l_mi-r_mi)/2
    Peak_val = numpy.max(y[int(a):int(b)])
    Peak_t = list(y[int(a):int(b)]).index(Peak_val)+a
    ampl = Peak_val - Base_val

    #compute areas
    xx = numpy.arange(int(a),int(b),1)
    yy = y[int(a):int(b)]
    yy2 = abs(y[int(a):int(b)])+abs(Base_val)
    h = auc(xx,yy)
    i = auc(xx,yy2)
    
    return numpy.array([a,b,lenght,Base_val,Peak_t,Peak_val,ampl,l_derivada,r_derivada,h,i,start_val,end_val])

def get_greenness(series,start,minimum_up):


    """
    
    This function get info about the browness of a cycle.
    
    Reference: 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        start : integer 
            Position along time series axis
        minimum_up : float 
            Minimum growing that will be used to detect a new cycle.

    Returns
    -------
    numpy.ndarray:
        array of peaks
    """
    
    #funtion to get the start/end of interval
    idx = numpy.where(series == numpy.amin(series))[0][0]
    series = series[idx:]
    l_mi = min(series)
    ma = max(series)
    dis = ma-l_mi

    #get cummulative sum from left side
    ret = ((series - l_mi)/ma)
    xp = numpy.arange(0,len(ret))    
    c = numpy.interp(1-minimum_up, ret, xp) + start 
    a = numpy.interp(minimum_up, ret, xp) + start
    
    return a,c,l_mi

def get_brownness(series,start,minimum_up):

    """
    
    This function get info about the browness of a cycle.
    
    Reference: 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        start : integer 
            Position along time series axis
        minimum_up : integer 
            Minimum growing that will be used to detect a new cycle.

    Returns
    -------
    numpy.ndarray:
        array of peaks
    """
    
    #funtion to get the start/end of interval  
    idx = numpy.where(series == numpy.amin(series))[0][0]
    
    series = series[:idx]

    r_mi = min(series)
    ma = max(series)
    dis = ma-r_mi

    #get cummulative sum from left side
    series = series[::-1]
    
    ret = (series - r_mi)/ma
    xp = numpy.arange(0,len(ret))    

    d = start + idx - numpy.interp(1-minimum_up, ret, xp) + 1
    b = start + idx - numpy.interp(minimum_up, ret, xp) + 1
    
    return b,d,r_mi

def get_peaks(timeseries,minimum_up):
    
    """
    
    This function find the peaks of a time series.
    
    Reference: 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        minimum_up : float
            Minimum growing that will be used to detect a new cycle.
    Returns
    -------
    numpy.ndarray:
        array of peaks
    """
    

    b = (numpy.diff(numpy.sign(numpy.diff(timeseries))) > 0).nonzero()[0] +1# local min
    c = (numpy.diff(numpy.sign(numpy.diff(timeseries))) < 0).nonzero()[0] +1# local max

    picos = []
    for vale, pic in zip(b,c):
        
        if abs(timeseries[vale] - timeseries[pic]) <= minimum_up:
            continue
        else:
            picos.append(vale)
            picos.append(pic)

    return numpy.sort(numpy.asarray(picos))


def domain(y,peaks,minimum_up):

    """
    
    This function find the cycles inside a timeseries.
    
    Reference: 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        peaks : list
            Peaks that are detected using function get_peaks.
        minimum_up : float
            Minimum growing that will be used to detect a new cycle.

    Returns
    -------
    pandas.dataframe:
        dataframe with the phenometrics
    """
    
    
    header_timesat=["Start","End","Length","Base val.","Peak t.","Peak val.","Ampl.","L. Deriv.","R. Deriv.","L.integral","S.integral","Start val.","End val."]
    dfp = pandas.DataFrame(columns=header_timesat)
    
    pos = 0
    
    for peak in range(1,len(peaks)-1):
        
        start = peaks[peak-1]
        midle = peaks[peak]
        end = peaks[peak+1]
        
        if y[midle] < y[start] or y[midle] < y[end]:
            continue
            
        elif (y[midle] - y[start]) <= minimum_up:
            continue    
           
        elif (y[midle] - y[end]) <= minimum_up :
            
            cond = (y[midle] - y[end]) >= minimum_up
            pp=peak+1
            
            while cond != True and pp <= peaks.shape[0]-1:
                end = peaks[pp]
                cond = (y[midle] - y[end]) > minimum_up
                pp = pp+1

        series = y[start:midle]    
         
        if series.shape[0] <= 2:
            continue
                
        #get left side
        a,c,l_mi = get_greenness(series,start,minimum_up)

        #get right side
        series = None
        series = y[midle:end]
                
        if series.shape[0] <= 2:
            continue
                
        b,d,r_mi = get_brownness(series,midle,minimum_up)
    
        dfp.loc[pos] = get_ts_metrics(y,a,c,l_mi,b,d,r_mi,minimum_up)
        
        dfp = dfp.dropna()        

        pos +=1
                    
    return dfp

def pheno(time_series, minimum_up, treshold = 0.125, window = 7, show=True):
    
    """
    
    This function get the phenological metrics.
    
    Reference: 

    Keyword arguments:
        timeseries : numpy.ndarray
            Your time series.
        minimum_up : float
            Minimum growing that will be used to detect a new cycle.
        treshold: float, 0.125 default
            Maximum distance used
        Window: integer (odd number) default 7
            Size of the window used for filtering with the Savitzky-Golay 
        show: boolean
            This inform if you want to plot the series with the starting and ending of each cycle detected.
    Returns
    -------
    pandas.dataframe:
        dataframe with the phenometrics
    timeseries plot with the start and end points
    """
    
    #This function perform the filtering with the savitky-golay mehtod
    y = get_filtered_series(time_series,window,max_dist_fitting)
    
    #This function detect peaks on the timesries
    peaks = get_peaks(y,window,minimum_up) 
    
    #This functions detect cycles and compute phenometrics from each of them
    dfp = domain(y,peaks,minimum_up)
    
    #This show the plot with the timeseries
    if show == True:
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.plot(time_series,label='TimeSeries')
        plt.plot(y,label='SG-Phenometrics')
        ax.scatter(dfp['Start'],dfp['Start val.'],label='Start point',color='green')
        ax.scatter(dfp['End'],dfp['End val.'],label='End point',color='red')
        legend = ax.legend(loc='uper left', shadow=True, fontsize='large',ncol=5,bbox_to_anchor=(1,1.15))
        plt.show()
    
    return dfp