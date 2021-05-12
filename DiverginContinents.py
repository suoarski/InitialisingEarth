

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
resolution = 400
earthRadius = 6317

sphereMesh = pv.Sphere(radius=earthRadius, theta_resolution=resolution, phi_resolution=resolution)
sphereXYZ, sphereFaces = sphereMesh.points, sphereMesh.faces




#Get pygplates to resolve data at specified time
boundaryXYZ, boundaryType, sharedBoundID = [], [], []
resolvedTopologies, sharedBoundarySections = [], []
pygplates.resolve_topologies(platePolygonsDirectory, rotationsDirectory, resolvedTopologies, int(time), sharedBoundarySections)

#Loop through shared boundary sections and subsections
for i, shareBound in enumerate(sharedBoundarySections):
    boundType = shareBound.get_feature().get_feature_type()
    isOceanicRidge = boundType == pygplates.FeatureType.gpml_mid_ocean_ridge
    
    #shareBoundXYZ = []
    for sharedSubSection in shareBound.get_shared_sub_segments():
        latLon = sharedSubSection.get_resolved_geometry().to_lat_lon_array()
        lon, lat = latLon[:, 1], latLon[:, 0]
        XYZ = earthInit.lonLatToCartesian(lon, lat)
        
        #shareBoundXYZ.append()
        
        #Append data to lists
        #sharedPlateIds = [i.get_resolved_feature().get_reconstruction_plate_id() for i in sharedSubSection.get_sharing_resolved_topologies()]
        for xyz in XYZ:
            boundaryXYZ.append(xyz)
            sharedBoundID.append(i)
            if isOceanicRidge:
                boundaryType.append(1)
            else:
                boundaryType.append(0)

boundaryXYZ, boundaryType = np.array(boundaryXYZ), np.array(boundaryType)
boundaryXYZ *= earthRadius

#distToBound, distToBoundID = KDTree(boundaryXYZ).query(sphereXYZ)
#nearbyBoundType = boundaryType[distToBoundID]

'''
nextID = distToBoundID + 1
nextID[nextID==np.max(nextID)] = 0
prevID = distToBoundID - 1
prevID[prevID==-1] = np.max(distToBoundID)
'''
boundaryId = np.arange(boundaryXYZ.shape[0])

plotter = pv.Plotter()
lineIds, lineCentres = [], []
for i in np.unique(sharedBoundID):
    sharesThis = (i==sharedBoundID)
    idx = boundaryId[sharesThis]
    for line in np.array([idx[:-1], idx[1:]]).T.astype(int):
        lineIds.append(line)
        lineCentres.append(np.mean((boundaryXYZ[line[0]], boundaryXYZ[line[1]]), axis=0))
    
    lines = pyvistaLinesFromPoints(boundaryXYZ[idx])
    plotter.add_mesh(lines, color='b')
lineIds = np.array(lineIds)
lineCentres = np.array(lineCentres)

distToLines, distToLineIds = KDTree(lineCentres).query(sphereXYZ)
nearbyLinePoints = boundaryXYZ[lineIds[distToLineIds]]

distToBound = []
for i, xyz in enumerate(sphereXYZ):
    linePoints = nearbyLinePoints[i]
    #numerator = np.linalg.norm(np.cross(xyz - linePoints[0], xyz - linePoints[1]))
    #denominator = np.linalg.norm(linePoints[1] - linePoints[0])
    numerator = np.linalg.norm(np.cross(linePoints[1] - xyz, linePoints[1] - linePoints[0]))
    denominator = np.linalg.norm(linePoints[1] - linePoints[0])
    distToLine = numerator / denominator
    distToZero = np.linalg.norm(linePoints[0] - xyz)
    distToOne = np.linalg.norm(linePoints[1] - xyz)
    
    v = linePoints[1] - linePoints[0]
    w = xyz - linePoints[0]
    if np.dot(w, v) <= 0:
        distToBound.append(distToZero)
    elif np.dot(v, v) <= np.dot(w, v):
        distToBound.append(distToOne)
    else:
        distToBound.append(distToLine)
    
    
    #distToBound.append(numerator / denominator)
distToBound = np.array(distToBound)

print(distToBound)
print(nearbyLinePoints.shape)
print(sphereXYZ.shape)




heightMap = np.zeros(sphereXYZ.shape[0])
heightMap -= 400 / (1 + np.exp(distToBound/200))

x = np.arange(0, 1000, 0.1)
y = 1 / (1 + np.exp(x/200))
plt.plot(x, y)
#plt.plot(distToBound)
#plt.show()

newSphereXYZ = earthInit.setRadialComponent(sphereXYZ, earthRadius+heightMap)
newSphereMesh = pv.PolyData(newSphereXYZ, sphereFaces)

        
#plotter = pv.Plotter()
plotter.add_mesh(newSphereMesh, scalars=heightMap)
plotter.add_mesh(lineCentres)#, scalars=sharedBoundID)
plotter.show()

#print(nearbyBoundType[:10])
#nearbyBoundType = np.sum(boundaryType[distToBoundID], axis=1)
