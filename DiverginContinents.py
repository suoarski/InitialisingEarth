

import os
import sys
sys.path.insert(1, 'C:/Users/BGH360/Desktop/KilianLiss/UniSydney/Research/20210208MovingPlates/pyGplates/pygplates_rev28_python38_win64')
import pygplates

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import InitialisingEarth as earthInit

#Creates lines as pyvista mesh
def pyvistaLinesFromPoints(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

#Directory for data files
platePolygonsDirectory = './Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies_PMAG.gpmlz'
platePolygonsDirectory400MYA = './Matthews_etal_GPC_2016_Paleozoic_PlateTopologies_PMAG.gpmlz'
rotationsDirectory = './Matthews_etal_GPC_2016_410-0Ma_GK07_PMAG.rot'
coastLinesDirectory = './Matthews_etal_GPC_2016_Coastlines.gpmlz'

#Parameters to be used
time = 5
resolution = 100
earthRadius = 6317

#Initiate earth as a sphere
sphereMesh = pv.Sphere(radius=earthRadius, theta_resolution=resolution, phi_resolution=resolution)
sphereXYZ, sphereFaces = sphereMesh.points, sphereMesh.faces

#Initiate lists to be used for boundary data storage
boundaryXYZ, boundaryType, sharedBoundID = [], [], []
resolvedTopologies, sharedBoundarySections = [], []
pygplates.resolve_topologies(platePolygonsDirectory, rotationsDirectory, resolvedTopologies, int(time), sharedBoundarySections)

#Loop through shared boundary sections and subsections
for i, shareBound in enumerate(sharedBoundarySections):
    boundType = shareBound.get_feature().get_feature_type()
    isOceanicRidge = (boundType == pygplates.FeatureType.gpml_mid_ocean_ridge)
    
    #Loop through shared subsegments and convert data to cartesian coordinates
    for sharedSubSection in shareBound.get_shared_sub_segments():
        latLon = sharedSubSection.get_resolved_geometry().to_lat_lon_array()
        lon, lat = latLon[:, 1], latLon[:, 0]
        XYZ = earthInit.lonLatToCartesian(lon, lat)
        
        #Append relevant data to lists
        for xyz in XYZ:
            boundaryXYZ.append(xyz)
            sharedBoundID.append(i)
            if isOceanicRidge:
                boundaryType.append(1)
            else:
                boundaryType.append(0)

#Convert lists to arrays
boundaryXYZ, boundaryType  = np.array(boundaryXYZ), np.array(boundaryType)
boundaryIndex = np.arange(boundaryXYZ.shape[0])
boundaryXYZ *= earthRadius

#For each line on plate boundaries, find the centres and create a list of lines by specifying the id of vertices that span it
plotter = pv.Plotter()
lineCentres, boundLines = [], []
for i in np.unique(sharedBoundID):
    sharesThis = (i==sharedBoundID)
    idx = boundaryIndex[sharesThis]
    bLines = np.array([idx[:-1], idx[1:]]).T.astype(int)
    if np.all(boundaryType[sharesThis] == 1):
        for line in bLines:
            boundLines.append(line)
            lineCentres.append(np.mean((boundaryXYZ[line[0]], boundaryXYZ[line[1]]), axis=0))
    
    #Add lines to pyvista plot
    lines = pyvistaLinesFromPoints(boundaryXYZ[idx])
    plotter.add_mesh(lines, color='b')

#Convert lists to arrays
boundLines = np.array(boundLines)
lineCentres = np.array(lineCentres)
#divergLines = boundLines
#divergLineCentres = lineCentres

#Create a list of line segments represented by 2 xyz vertex coordinates
distToLines, distToLinesIds = KDTree(lineCentres).query(sphereXYZ)
lineSegmentXYZ = boundaryXYZ[boundLines[distToLinesIds]]

#Find the distance of each sphere vertex from the plate boundaries
distToBound = []
for i, xyz in enumerate(sphereXYZ):
    lineXYZ = lineSegmentXYZ[i]
    
    #Append distance from vertex 0
    v = lineXYZ[1] - lineXYZ[0]
    w = xyz - lineXYZ[0]
    if np.dot(w, v) <= 0:
        distToZero = np.linalg.norm(lineXYZ[0] - xyz)
        distToBound.append(distToZero)
    
    #Append distance from vertex 1  
    elif np.dot(v, v) <= np.dot(w, v):
        distToOne = np.linalg.norm(lineXYZ[1] - xyz)
        distToBound.append(distToOne)
    
    #Append distance from somewhere in the line centre
    else:
        numerator = np.linalg.norm(np.cross(lineXYZ[1] - xyz, lineXYZ[1] - lineXYZ[0]))
        denominator = np.linalg.norm(lineXYZ[1] - lineXYZ[0])
        distToLine = numerator / denominator
        distToBound.append(distToLine)
distToBound = np.array(distToBound)

#Initiate a heightmap a lower points nearby boundaries
heightMap = np.zeros(sphereXYZ.shape[0])
heightMap -= 400 / (1 + np.exp(distToBound/100))

'''
x = np.arange(0, 1000, 0.1)
y = 1 / (1 + np.exp(x/200))
plt.plot(x, y)
plt.plot(distToBound)
plt.show()
'''

#Create a mesh of the final terrain
newSphereXYZ = earthInit.setRadialComponent(sphereXYZ, earthRadius+heightMap)
newSphereMesh = pv.PolyData(newSphereXYZ, sphereFaces)

#Plot the mesh
plotter.add_mesh(newSphereMesh, scalars=heightMap)
plotter.add_mesh(lineCentres)#, scalars=sharedBoundID)
plotter.show()

