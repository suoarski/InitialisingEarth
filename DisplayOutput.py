
import os
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
import InitialisingEarth as earthInit

#Specify options here
showPlateBounds = True
earthRadius = 6371

#Specify folder to read data from
folderName = 'run2'
dataDirectory = './OutputXYZData/{}'.format(folderName)
fileNameFormat = './OutputXYZData/{}/Time{}.csv'

#Directory for data files
platePolygonsDirectory = './Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies_PMAG.gpmlz'
platePolygonsDirectory400MYA = './Matthews_etal_GPC_2016_Paleozoic_PlateTopologies_PMAG.gpmlz'
rotationsDirectory = './Matthews_etal_GPC_2016_410-0Ma_GK07_PMAG.rot'
coastLinesDirectory = './Matthews_etal_GPC_2016_Coastlines.gpmlz'

#Based on the files in the data folder, we get a list of all available times that we have data for
availableTimes = []
for fileName in os.listdir(dataDirectory):
    if fileName.endswith('.csv'):
        availableTimes.append(float(fileName[4:-4]))
availableTimes = np.array(availableTimes)
timeRange = [np.min(availableTimes), np.max(availableTimes)]

#Get mesh data from file at specified time
def getMeshAtTime(time):
    t = availableTimes[KDTree(availableTimes.reshape(-1, 1)).query(time)[1]]
    fileDir = fileNameFormat.format(folderName, int(t))
    data = np.array([line.split(',') for line in open(fileDir).read().split('\n')[1:-1]]).T.astype(float)
    XYZ, Lat, Lon, heightMap = data[:3].T, data[3], data[4], data[5]

    #Create mesh for plotting
    faces = pv.PolyData(np.stack((Lat, Lon, np.zeros(len(Lat)))).T).delaunay_2d().faces
    sphereMesh = pv.PolyData(XYZ, faces)
    scallars = np.copy(heightMap)
    return sphereMesh, scallars, XYZ

#Class for an interactive pyvista plot
class earthRenderer():
    def __init__(self, time):
        self.kwargs = {'time': 5}
        self.plotter = pv.Plotter()
        
        #Add main mesh
        self.mesh, self.scallars, XYZ = getMeshAtTime(time)
        self.plotter.add_mesh(self.mesh, scalars=self.scallars, cmap='gist_earth')
        
        #Add boundary mesh
        boundaryData = earthInit.getPlateBoundaryData(time, platePolygonsDirectory, rotationsDirectory, earthRadius)
        self.boundActor = self.plotter.add_mesh(boundaryData[0], color='b')
        
        #Add slider to specify time with
        self.plotter.add_slider_widget(
            callback = lambda value: self('time', value),
            rng = timeRange,
            value = 5,
            title = 'Time',
            style = 'modern'
        )
        
    #Function to be called by slider updates
    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()
    
    #Update when slider changes values
    def update(self):
        time = self.kwargs['time']
        sphereMesh, scallars, XYZ = getMeshAtTime(time)
        self.plotter.update_coordinates(XYZ, mesh=self.mesh)
        self.plotter.update_scalars(scallars, mesh=self.mesh)
        self.plotter.remove_actor(self.boundActor)
        boundaryData = earthInit.getPlateBoundaryData(time, platePolygonsDirectory, rotationsDirectory, earthRadius)
        self.boundActor = self.plotter.add_mesh(boundaryData[0], color='b')
        return

#Initiate an instance of the earth renderer
earth = earthRenderer(availableTimes[-1])
earth.plotter.show()

