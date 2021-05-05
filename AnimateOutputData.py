
import os
import sys
sys.path.insert(1, 'C:/Users/BGH360/Desktop/KilianLiss/UniSydney/Research/20210208MovingPlates/pyGplates/pygplates_rev28_python38_win64')
import pygplates

import numpy as np
import pyvista as pv
import InitialisingEarth as init

#Function for creating vectors (arrows) representing speed of plate collisions
def createConvergingSpeedGlyph(time, rotationsDirectory, platePolygonsDirectory):
    rotationModel = pygplates.RotationModel(rotationsDirectory)
    boundaryData = init.getPlateBoundaryData(time, platePolygonsDirectory, rotationModel, earthRadius)
    SubZOneXYZ = boundaryData[1]

    subZoneMesh = pv.PolyData(SubZOneXYZ * 1.05)
    overPlateId, underPlateId = boundaryData[4], boundaryData[5]
    speeds = [init.getSpeed(xyz, rotationModel, time+1, time, underPlateId[i], overPlateId[i]) for i, xyz in enumerate(SubZOneXYZ)]

    subZoneMesh.point_arrays['speed'] = speeds
    subZoneMesh.point_arrays['orientation'] = SubZOneXYZ
    return subZoneMesh.glyph(geom=pv.Arrow(scale=10, direction=(1.0, 0.0, 0.0)), scale='speed', orient="orientation")

#Directories for pygplates data files
platePolygonsDirectory = './Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies_PMAG.gpmlz'
rotationsDirectory = './Matthews_etal_GPC_2016_410-0Ma_GK07_PMAG.rot'

#Specify properties and folder with data to animate
folderName = 'run2'
folderDir = './OutputXYZData/{}'.format(folderName)
earthRadius = 6371
showCollisionSpeed = True

#Create list of available times based on data files
times = []
for fileName in os.listdir(folderDir):
    if fileName.endswith('.csv'):
        times.append(int(fileName[4:-4]))
times.sort(reverse=True)


#Iterate through the available time steps and animate data
for i, time in enumerate(times):
    
    #Read data file
    fileDir = folderDir + '/Time{}.csv'.format(time)
    data = np.array([line.split(',') for line in open(fileDir).read().split('\n')[1:-1]]).T.astype(float)
    XYZ, Lat, Lon, heightMap = data[:3].T, data[3], data[4], data[5]

    #Create mesh for plotting
    faces = pv.PolyData(np.stack((Lat, Lon, np.zeros(len(Lat)))).T).delaunay_2d().faces
    sphereMesh = pv.PolyData(XYZ, faces)
    scallars = np.copy(heightMap)
    print(time)
    
    #Initiate plotter objects on first run of this loop
    if (i == 0):
        plotter = pv.Plotter()
        if showCollisionSpeed:
            subGlyph = createConvergingSpeedGlyph(time, rotationsDirectory, platePolygonsDirectory)
            boundAct = plotter.add_mesh(subGlyph, color='b')
        sphre = plotter.add_mesh(sphereMesh, scalars=scallars, cmap='gist_earth')
        plotter.add_scalar_bar()
        plotter.add_axes(interactive=True)
        print('Please select viewing orientation and then press q to itterate over animation frames')
        plotter.show(auto_close=False, window_size=[800, 608])
        plotter.open_movie('TectonicSimulation.mp4')
        plotter.write_frame()
    
    #Update plot on later runs of this loop
    else:
        plotter.remove_actor(sphre)
        sphre = plotter.add_mesh(sphereMesh, scalars=scallars, cmap='gist_earth')
        if showCollisionSpeed:
            plotter.remove_actor(boundAct)
            subGlyph = createConvergingSpeedGlyph(time, rotationsDirectory, platePolygonsDirectory)
            boundAct = plotter.add_mesh(subGlyph, color='b')
        plotter.write_frame()



