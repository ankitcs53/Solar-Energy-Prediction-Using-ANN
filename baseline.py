"""
This file preProcesses to form training data( years 1994 to 2005)
Authors: Ankit Choudhary, Vishal Chauhan
Description: Reads Mesonet and GEFS data and applies Catmull-Rom splines to interpolate GEFS data to Mesonet sites.
             Applies Data Segmentation to obtain season wise data segments
             Does it for each weather variable to obtain the 2-d matrix 
PS : mehtods buildSplines, getGrid, getDailyMeanSumGrid are provided as a part of the contest
"""

from netCDF4 import Dataset
import numpy as np
import math

dataPath = "/home/ankit/Documents/Deep Learning/Dataset/"

def loadMesonetData(filename,stationFilename="station_info.csv"):
    """
    loadMesonetData(filename,stationFilename)
    Description: loads Mesonet data and station data
    Parameters:
    filename (str) - Name of Mesonet csv file being read.
    stationData (str) - Name of file containing station information. Default station_info.csv.
    Returns:
    data - numpy array containing total daily solar radiation for each date
    dates - numpy array of dates as integers in YYYYMMDD format
    stationData - numpy structured array containing the station information, including lat-lon and elevation.
    """
    data = np.genfromtxt(filename,delimiter=',',skip_header=1,dtype=float)
    dates = np.array(data[:,0].T,dtype=int)
    data = data[:,1:]
    stationData = np.genfromtxt(stationFilename,delimiter=',',dtype=[("stid","S4"),("nlat",float),("elon",float),("elev",float)],skip_header=1)
    return data,dates,stationData

def loadData(filename):
    """
    loadData()
    Description: Creates a netCDF4 file object for the specified file.
    Parameters:
    filename (str) - name of the GEFS netCDF4 file.
    Returns:
    data - Dataset object that allows access of GEFS data.
    """
    data = Dataset(filename)
    return data

def getGrid(data,date,fHour,eMember):
    """
    getGrid()
    Description: Load GEFS data from a specified date, forecast hour, and ensemble member.
    Parameters:
    data (Dataset) - Dataset object from loadData
    date (int) - date of model run in YYYYMMDD format
    fHour (int) - forecast hour
    eMember (int) - ensemble member id.
    Returns: numpy 2d array from the specified output
    """
    dateIdx = np.where(data.variables['intTime'][:] == date)[0]
    fIdx = np.where(data.variables['fhour'][:] == fHour)[0]
    eIdx = np.where(data.variables['ens'][:] == eMember)[0]
    return data.variables.values()[-1][dateIdx,eIdx,fIdx][0]

def getDailyMeanSumGrid(data,date):
    """
    getDailyMeanSumGrid()
    Description: For a particular date, sums over all forecast hours for each ensemble member then takes the 
    mean of the summed data and scales it by the GEFS time step.
    Parameters:
    data (Dataset) - netCDF4 object from loadData
    date (int) - date of model run in YYYYMMDD format
    Returns - numpy 2d array from the specified output
    """
    dateIdx = np.where(data.variables['intTime'][:] == date)[0]
    fIdx = np.where(data.variables['fhour'][:] <= 24)[0]
    return data.variables.values()[-1][dateIdx,:,fIdx,:,:].sum(axis=2).mean(axis=1)[0]*3600*3

def buildSplines(data,grid,stationdata):
    """
    buildSplines()
    Description: For each station in stationdata, a set of Catmull-Rom splines are calculated to interpolate from the
    nearest grid points to the station location. A set of horizontal splines are created at each latitude and 
    interpolated at the station longitude. Then another spline is built from the output of those splines to get the 
    value at the station location.
    Paramters:
    data (Dataset) - netCDF4 object with the GEFS data 
    grid (numpy array) - the grid being interpolated
    stationdata (numpy structured array) - array containing station names, lats, and lons.
    Returns: array with the interpolated values.
    """
    outdata=np.zeros(stationdata.shape[0])
    #print np.zeros(stationdata.shape[0])
    for i in xrange(stationdata.shape[0]):
        slat,slon=stationdata['nlat'][i],stationdata['elon'][i]
        nearlat=np.where(np.abs(data.variables['lat'][:]-slat)<2)[0]
        nearlon=np.where(np.abs(data.variables['lon'][:]-slon-360)<2)[0]
        Spline1=np.zeros(nearlon.shape)

        for l,lat in enumerate(nearlat):
            Spline1[l]=Spline(grid[nearlat[l],nearlon],(slon-np.floor(slon))/1)
        outdata[i]=Spline(Spline1,(slat-np.floor(slat))/1)

    #print outdata
    return outdata

def Spline(y,xi):
    """
    Spline
    Description: Given 4 y values and a xi point, calculate the value at the xi point.
    Parameters:
    y - numpy array with 4 values from the 4 nearest grid points
    xi - index at which the interpolation is occurring.
    Returns: yi - the interpolated value
    """
    return 0.5*((2*y[1])+(y[2]-y[0])*xi+(-y[3]+4*y[2]-5*y[1]+2*y[0])*xi**2+(y[3]-3*y[2]+3*y[1]-y[0])*xi**3)

def doy(day,month,year):
    N1 = math.floor(275 * month / 9)
    N2 = math.floor((month + 9) / 12)
    N3 = (1 + math.floor((year - 4 * math.floor(year / 4) + 2) / 3))
    N = N1 - (N2 * N3) + day - 30
    return N

def SpacialInterpolate(filename):
    MesonetData,dates,stationdata = loadMesonetData(dataPath + 'train.csv',dataPath + "station_info.csv")

    data = loadData(dataPath + filename)

    output1 = []
    output2 = []
    output3 = []
    output4 = []
    for date in dates:
    #    print date*100
        if date >= 20060000:
            break
        da = date % 100
        yr = (date - (date % 10000) ) / 10000
        mo = ( date - (yr * 10000) ) / 100
        doyr = doy(da,mo,yr)

        grid = getDailyMeanSumGrid(data,date*100)
        outdata=buildSplines(data,grid,stationdata)
        
        if doyr <= 90 :
            output1.extend(outdata.tolist())
        elif doyr <= 180:
            output2.extend(outdata.tolist())
        elif doyr <= 270:
            output3.extend(outdata.tolist())
        else:
            output4.extend(outdata.tolist())
        
    return (np.array(output1) , np.array(output2) , np.array(output3) , np.array(output4))

def getOutput():

    values = np.genfromtxt(dataPath+"train.csv",delimiter=',',skip_header=1,dtype=float)

    siz = values.shape[0]

    output1 = []
    output2 = []
    output3 = []
    output4 = []


    for i in range(0,siz):
        date = int(values[i,0])

        if date >= 20060000:
            break

        da = date % 100
        yr = (date - (date % 10000) ) / 10000
        mo = ( date - (yr * 10000) ) / 100
        doyr = doy(da,mo,yr)

        if doyr <= 90 :
            output1.extend(values[i,1:].tolist())
        elif doyr <= 180:
            output2.extend(values[i,1:].tolist())
        elif doyr <= 270:
            output3.extend(values[i,1:].tolist())
        else:
            output4.extend(values[i,1:].tolist())

    return (np.array(output1) , np.array(output2) , np.array(output3) , np.array(output4))
        
def ProcessData():

    output1 = SpacialInterpolate("apcp_sfc_latlon_subset_19940101_20071231.nc")
    output2 = SpacialInterpolate("dlwrf_sfc_latlon_subset_19940101_20071231.nc")
    output3 = SpacialInterpolate("dswrf_sfc_latlon_subset_19940101_20071231.nc")
    output4 = SpacialInterpolate("pres_msl_latlon_subset_19940101_20071231.nc")
    output5 = SpacialInterpolate("pwat_eatm_latlon_subset_19940101_20071231.nc")
    output6 = SpacialInterpolate("spfh_2m_latlon_subset_19940101_20071231.nc")
    output7 = SpacialInterpolate("tcdc_eatm_latlon_subset_19940101_20071231.nc")
    output8 = SpacialInterpolate("tcolc_eatm_latlon_subset_19940101_20071231.nc")
    output9 = SpacialInterpolate("tmax_2m_latlon_subset_19940101_20071231.nc")
    output10 = SpacialInterpolate("tmin_2m_latlon_subset_19940101_20071231.nc")
    output11 = SpacialInterpolate("tmp_2m_latlon_subset_19940101_20071231.nc")
    output12 = SpacialInterpolate("tmp_sfc_latlon_subset_19940101_20071231.nc")
    output13 = SpacialInterpolate("ulwrf_sfc_latlon_subset_19940101_20071231.nc")
    output14 = SpacialInterpolate("ulwrf_tatm_latlon_subset_19940101_20071231.nc")
    output15 = SpacialInterpolate("uswrf_sfc_latlon_subset_19940101_20071231.nc")


    #get prediction values
    values = getOutput()

    outputnn1 = np.column_stack((output1[0],output2[0],output3[0],output4[0],output5[0],output6[0],output7[0],output8[0],output9[0],output10[0],output11[0],output12[0],output13[0],output14[0],output15[0],values[0]))
    outputnn2 = np.column_stack((output1[1],output2[1],output3[1],output4[1],output5[1],output6[1],output7[1],output8[1],output9[1],output10[1],output11[1],output12[1],output13[1],output14[1],output15[1],values[1]))
    outputnn3 = np.column_stack((output1[2],output2[2],output3[2],output4[2],output5[2],output6[2],output7[2],output8[2],output9[2],output10[2],output11[2],output12[2],output13[2],output14[2],output15[2],values[2]))
    outputnn4 = np.column_stack((output1[3],output2[3],output3[3],output4[3],output5[3],output6[3],output7[3],output8[3],output9[3],output10[3],output11[3],output12[3],output13[3],output14[3],output15[3],values[3]))
    
    np.savetxt(dataPath+"inputnn1.csv",outputnn1,delimiter=",")
    np.savetxt(dataPath+"inputnn2.csv",outputnn2,delimiter=",")
    np.savetxt(dataPath+"inputnn3.csv",outputnn3,delimiter=",")
    np.savetxt(dataPath+"inputnn4.csv",outputnn4,delimiter=",")



def main():
    """ MesonetData,dates,stationdata = loadMesonetData(dataPath + 'train.csv',dataPath + "station_info.csv")
    data = loadData(dataPath + "dswrf_sfc_latlon_subset_19940101_20071231.nc")
    lons = data.variables['lon'][:]
    lats = data.variables['lat'][:]
    f = open('spline_submission.csv', 'w')
    header = ["Date"]
    header.extend(stationdata['stid'].tolist())
    
    f.write(",".join(header) + "\n")

        
    for date in dates:
        print date*100
        grid = getDailyMeanSumGrid(data,date*100)
        outdata=buildSplines(data,grid,stationdata)
        output = np.concatenate((output,outdata),axis=0)
        f.write("%d" % date + ",")
        np.savetxt(f, outdata.T, delimiter=',',fmt='%7.0f')      
    
    f.close()
    data.close()
    """

    ProcessData()

if __name__ == "__main__":
    main()