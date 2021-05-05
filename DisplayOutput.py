
import os
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

#Specify folder to read data from
folderName = 'run1'
dataDirectory = './OutputXYZData/{}'.format(folderName)
fileNameFormat = './OutputXYZData/{}/Time{}.csv'

#Based on the file in the data folder, we get a list of all available times that we have data for
availableTimes = []
for fileName in os.listdir(dataDirectory):
    if fileName.endswith('.csv'):
        availableTimes.append(float(fileName[4:-4]))
availableTimes = np.array(availableTimes)
timeRange = [np.min(availableTimes), np.max(availableTimes)]

#This function will be called each time the slider moves
plotter = pv.Plotter()
def createMesh(time):
    
    #Read data from file
    t = availableTimes[KDTree(availableTimes.reshape(-1, 1)).query(time)[1]]
    fileDir = fileNameFormat.format(folderName, int(t))
    data = np.array([line.split(',') for line in open(fileDir).read().split('\n')[1:-1]]).T.astype(float)
    XYZ, Lat, Lon, heightMap = data[:3].T, data[3], data[4], data[5]
    
    #Create mesh for plotting
    faces = pv.PolyData(np.stack((Lat, Lon, np.zeros(len(Lat)))).T).delaunay_2d().faces
    sphereMesh = pv.PolyData(XYZ, faces)
    scallars = np.copy(heightMap)
    
    #Update plot
    if plotter.mesh == None:
        plotter.add_mesh(sphereMesh, scalars=scallars, cmap='gist_earth')
    else:
        plotter.update_coordinates(XYZ)
        plotter.update_scalars(scallars)
    return

#Add slider widget with 'createMesh' as it's callback function, then show plot
plotter.add_slider_widget(createMesh, timeRange, title='Time')
plotter.show()
