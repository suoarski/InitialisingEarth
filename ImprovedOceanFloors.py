
import os
import sys
sys.path.insert(1, 'C:/Users/BGH360/Desktop/KilianLiss/UniSydney/Research/20210208MovingPlates/pyGplates/pygplates_rev28_python38_win64')
import pygplates

import numpy as np
import pyvista as pv
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import InitialisingEarth as earthInit
from sklearn.cluster import DBSCAN
from scipy.interpolate import *

#Directory for data files
platePolygonsDirectory = './Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies_PMAG.gpmlz'
platePolygonsDirectory400MYA = './Matthews_etal_GPC_2016_Paleozoic_PlateTopologies_PMAG.gpmlz'
rotationsDirectory = './Matthews_etal_GPC_2016_410-0Ma_GK07_PMAG.rot'
coastLinesDirectory = './Matthews_etal_GPC_2016_Coastlines.gpmlz'
rotationModel = pygplates.RotationModel(rotationsDirectory)

#To calculate the ocean floor, we need the speed that plates are diverging at
#(oceanicRidgeXYZ, timeFrom, timeTo, rotationModel, sharedPlateIds)
def getRidgeSpreadingSpeeds(time, deltaTime, rotationModel, ridgeXYZ, ridgeIds):
    timeFrom, timeTo = time+deltaTime, time
    spreadingSpeed = []
    for i, xyz in enumerate(ridgeXYZ):
        if len(ridgeIds[i]) == 2:
            spreadingSpeed.append(earthInit.getSpeed(xyz, rotationModel, timeFrom, timeTo, ridgeIds[i][0], ridgeIds[i][1]))
        else:
            spreadingSpeed.append(0.0)
    return np.array(spreadingSpeed)

#We use the griddata interpolation from the scipy library to remesh our sphere whilst maintianing heights
#def movePlatesAndRemeshSphere(sphereXYZ, heightMap, plateIds, time, deltaTime,  rotationModel, plateBoundsXYZ, boundIds):
def movePlatesAndRemeshSphere(sphereXYZ, heightMap, time, deltaTime, rotationModel, resolution, boundaryData):
    #(boundXYZ, subZoneXYZ, ridgeXYZ, boundPlateIds, overridingPlateIds, subductingPlateIds, ridgeIds, boundaryType, shareBoundID)
    sphereLon, sphereLat = earthInit.cartesianToLonLat(sphereXYZ[:, 0], sphereXYZ[:, 1], sphereXYZ[:, 2])
    pointFeatures = earthInit.createPointFeatures(sphereLon, sphereLat)
    plateIds = earthInit.getPlateIdsAtTime(time, rotationModel, pointFeatures, platePolygonsDirectory)
    movedSphereXYZ = earthInit.movePlates(rotationModel, sphereXYZ, plateIds, time, deltaTime)
    distToSubZone = KDTree(boundaryData[0]).query(movedSphereXYZ)[0]
    
    #Get longitude and latitude data
    sphereLonLat = np.stack(earthInit.cartesianToLonLat(sphereXYZ[:, 0], sphereXYZ[:, 1], sphereXYZ[:, 2]), axis=1)
    movedLonLat = np.stack(earthInit.cartesianToLonLat(movedSphereXYZ[:, 0], movedSphereXYZ[:, 1], movedSphereXYZ[:, 2]), axis=1)
    
    #When two plates collide into each other, some vertices of the two plates will overlap
    #In the case of subduction, the vertices of the mountains will be higher than those of the subducting plate
    #We want our interpolation algorithm to sample from the mountains, not the subducting plate
    #To do so, we use a clustering algorithm (DBSCAN) to check if vertices are closer than usual, 
    #and filter out vertices that are too far away from plate boundaries 
    
    #Use clustering algorithm to find vertices that are too close
    spacing = (150 / resolution)
    lonLatForClustering = np.copy(movedLonLat)
    lonLatForClustering[:, 0] /= 2
    cluster = DBSCAN(eps=spacing, min_samples=2, n_jobs=mp.cpu_count()).fit(lonLatForClustering)
    isCluster = (cluster.labels_ != -1)
    
    #Filter out vertices that are too far away from plate boundaries
    isCloseEnough = (distToSubZone < 300 * deltaTime * (np.cos(np.radians(movedLonLat[:, 1])) + 0.1))
    isOverlapping = isCloseEnough * isCluster
    
    #We set overlapping vertices to the height of its 6 nearest neighbours
    #So subducting vertices are now at mountain heights.
    spherePointsInCluster = movedSphereXYZ[isOverlapping]
    dists, ids = KDTree(movedSphereXYZ).query(spherePointsInCluster, k=6, workers=mp.cpu_count())
    neighbourHeights = heightMap[ids]
    maxNeighbourHeights = np.max(neighbourHeights, axis=1)
    heightMap[isOverlapping] = maxNeighbourHeights
    
    #The first griddata interpolation gives better results, but leaves NaNs behind
    #The second griddata interpolation is used to fill in the NaNs but with a slightly less accurate interpolation
    interpolatedHeights = griddata(movedLonLat, heightMap, sphereLonLat)
    whereNAN = np.argwhere(np.isnan(interpolatedHeights))
    interpolatedHeights[whereNAN] = griddata(movedLonLat, heightMap, sphereLonLat[whereNAN], method='nearest')
    return interpolatedHeights

#Parameters to be used
timeFrom, timeTo = 100, 0
deltaTime = 5
resolution = 100
earthRadius = 6317
times = np.arange(timeFrom, timeTo, - deltaTime)

maxDepth = 5.5
ageProportion = 0.35
ridgeDepth = 2.5

#Initiate earth as a sphere
sphereMesh = pv.Sphere(radius=earthRadius, theta_resolution=resolution, phi_resolution=resolution)
sphereXYZ, sphereFaces = sphereMesh.points, sphereMesh.faces
heightMap = np.zeros(len(sphereFaces))

#(boundXYZ, subZoneXYZ, ridgeXYZ, boundPlateIds, overridingPlateIds, subductingPlateIds, ridgeIds, boundaryType, shareBoundID)
boundaryData = earthInit.getPlateBoundaryData(timeFrom, platePolygonsDirectory, rotationsDirectory, earthRadius)
boundXYZ = boundaryData[0]
ridgeXYZ = boundaryData[2]
ridgeIds = boundaryData[6]
boundaryType = boundaryData[7]
shareBoundID = boundaryData[8]


#Initialize heightmap
distToRidge, closeRidgeIDs = earthInit.getDistanceToBoundary(sphereXYZ, boundXYZ, boundaryType, shareBoundID)
spreadingSpeed = getRidgeSpreadingSpeeds(timeFrom, deltaTime, rotationModel, ridgeXYZ, ridgeIds)
oceanFloorAge = np.abs(distToRidge) / (spreadingSpeed[closeRidgeIDs])
oceanDepth = - (ridgeDepth + ageProportion * oceanFloorAge**0.5)
oceanDepth[oceanDepth < -maxDepth] = -maxDepth
heightMap = oceanDepth

#Create mesh for plotting
sphereXYZwithHeights = earthInit.setRadialComponent(sphereXYZ, heightMap * 200 + earthRadius)
sphereMeshWithHeights = pv.PolyData(sphereXYZwithHeights, sphereFaces)

#Initiate plotter
plotter = pv.Plotter()
#boundAct = plotter.add_mesh(boundXYZ * 1.02)#, scalars=boundaryType)
plotter.add_mesh(sphereMeshWithHeights, scalars=heightMap)
#plotter.add_scalar_bar()
#plotter.add_axes(interactive=True)
print('Please select viewing orientation and then press q to itterate over animation frames')
plotter.show(auto_close=False, window_size=[800, 608])

#Run simulation through various times
for i, time in enumerate(times[1:]):
    
    #(boundXYZ, subZoneXYZ, ridgeXYZ, boundPlateIds, overridingPlateIds, subductingPlateIds, ridgeIds, boundaryType, shareBoundID)
    boundaryData = earthInit.getPlateBoundaryData(time, platePolygonsDirectory, rotationsDirectory, earthRadius)
    boundXYZ = boundaryData[0]
    ridgeXYZ = boundaryData[2]
    ridgeIds = boundaryData[6]
    boundaryType = boundaryData[7]
    shareBoundID = boundaryData[8]
    
    heightMap = movePlatesAndRemeshSphere(sphereXYZ, heightMap, time, deltaTime, rotationModel, resolution, boundaryData)
    print(np.max(heightMap))
    print(np.min(heightMap))
    #oceanFloorAge += deltaTime
    #heightMap[heightMap>-5.5] -= ageProportion * deltaTime / 2 * np.sqrt(oceanFloorAge[heightMap>-5.5])
    #heightMap[heightMap<-5.5] = -5.5
    
    
    sphereXYZwithHeights = earthInit.setRadialComponent(sphereXYZ, heightMap * 200 + earthRadius)
    #sphereMeshWithHeights = pv.PolyData(sphereXYZwithHeights, sphereFaces)
    
    #plotter.clear()
    #plotter.add_mesh(sphereMeshWithHeights, scalars=heightMap)
    plotter.update_coordinates(sphereXYZwithHeights)#, mesh=sphereMeshWithHeights)
    plotter.update_scalars(heightMap)#, mesh=sphereMeshWithHeights)
    #boundAct = plotter.add_mesh(boundaryData[0])
    #plotter.show(auto_close=False)
    plotter.show(auto_close=False)
    



