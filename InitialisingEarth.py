

import os
import sys
sys.path.insert(1, 'C:/Users/BGH360/Desktop/KilianLiss/UniSydney/Research/20210208MovingPlates/pyGplates/pygplates_rev28_python38_win64')

import pygplates
import numpy as np
import pyvista as pv
import multiprocessing as mp
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from perlin_noise import PerlinNoise
from scipy.interpolate import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance

from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import time as tme

#=============================Functions Related to Coordinate Transformations ============================================================================
#Coordinate transformation from spherical polar to cartesian
def polarToCartesian(R, Theta, Phi):
    X = R * np.cos(Theta) * np.sin(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Phi)
    return X, Y, Z

#Coordinate transformation from cartesian to polar
def cartesianToPolarCoords(X, Y, Z):
    R = (X**2 + Y**2 + Z**2)**0.5
    Theta = np.arctan2(Y, X)
    Phi = np.arccos(Z / R)
    return R, Theta, Phi

#Takes longatude and latitude coordinates and converts them to cartesian coordinates
def lonLatToCartesian(lon, lat, radius=1.0):
    X, Y, Z = polarToCartesian(radius, np.radians(lon+180), np.radians(90 - lat))
    return np.stack((X, Y, Z), axis=-1)

#Takes cartesian coordinates and converts them to longatude and latitude
def cartesianToLonLat(X, Y, Z):
    r, theta, phi = cartesianToPolarCoords(X, Y, Z)
    theta, phi = np.degrees(theta), np.degrees(phi)
    lon, lat = theta - 180, 90 - phi
    lon[lon < -180] = lon[lon < -180] + 360
    return lon, lat

#Function for moving vertices on a sphere along the radial direction by amount of delta radius (dr)
def moveAlongRadialDirection(XYZ, dr):
    r, theta, phi = cartesianToPolarCoords(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2])
    newSphereX = XYZ[:, 0] + np.cos(theta) * np.sin(phi) * dr
    newSphereY = XYZ[:, 1] + np.sin(theta) * np.sin(phi) * dr
    newSphereZ = XYZ[:, 2] + np.cos(phi) * dr
    return np.stack((newSphereX, newSphereY, newSphereZ), axis=-1)

#Given a sphere XYZ coordinates, we set the radius of all coordinates
def setRadialComponent(XYZ, r):
    rad, theta, phi = cartesianToPolarCoords(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2])
    newSphereX = np.cos(theta) * np.sin(phi) * r
    newSphereY = np.sin(theta) * np.sin(phi) * r
    newSphereZ = np.cos(phi) * r
    return np.stack((newSphereX, newSphereY, newSphereZ), axis=-1)

#Returns a rotation quaternion
def quaternion(axis, angle):
    return [np.sin(angle/2) * axis[0], 
            np.sin(angle/2) * axis[1], 
            np.sin(angle/2) * axis[2], 
            np.cos(angle/2)]

#Normalizes the height map or optionally brings the heightmap within specified height limits
def normalizeArray(A, minValue=0.0, maxValue=1.0):
    A = A - min(A)
    A = A / (max(A) - min(A))
    return A * (maxValue - minValue) + minValue

#Function for exporting terrain data to file
def writeXYZData(XYZ, heightMap, outFolder, time):
    Lon, Lat = cartesianToLonLat(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2])
    csvDirectory = outFolder + '/Time{}.csv'.format(time)
    with open(csvDirectory, 'w') as file:
        file.write('X, Y, Z, Longitude, Latitude, Actual Heights\n')
        for i, xyz in enumerate(XYZ):
            file.write('{}, {}, {}, {}, {}, {}\n'.format(xyz[0], xyz[1], xyz[2], Lon[i], Lat[i], heightMap[i]))

#Create directory to export our XYZ data into
def createXYZOutputDirectory(props, directory='./OutputXYZData'):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    runNumber = 1
    outputDirectory = directory + '/run{}'.format(runNumber)
    while os.path.isdir(outputDirectory):
        runNumber += 1
        outputDirectory = directory + '/run{}'.format(runNumber)
    os.mkdir(outputDirectory)
    
    #Create text file of properties used
    with open(outputDirectory+'/Properties.txt', 'w') as file:
        for k in props.keys():
            file.write('{}: {}\n'.format(k, str(props[k])))
    return outputDirectory

#========================================Code Related to pygplates ================================================================
#Creates a list of point features from our sphere's coordinates to be used by pygplates
def createPointFeatures(lon, lat):
    pointsOnSphere = [pygplates.PointOnSphere(float(lat[i]), float(lon[i])) for i in range(len(lon))]
    pointFeatures = []
    for point in pointsOnSphere:
        pointFeature = pygplates.Feature()
        pointFeature.set_geometry(point)
        pointFeatures.append(pointFeature)
    return pointFeatures

#Returns a list of plate Ids for points on our sphere
def getPlateIdsAtTime(time, rotationModel, pointFeatures, platePolygonsDirectory):
    assignedPointFeatures = pygplates.partition_into_plates(
        platePolygonsDirectory,
        rotationModel,
        pointFeatures,
        reconstruction_time=float(time),
        properties_to_copy = [
            pygplates.PartitionProperty.reconstruction_plate_id,
            pygplates.PartitionProperty.valid_time_period])
    featureIds = [feat.get_reconstruction_plate_id() for feat in assignedPointFeatures]
    return np.array(featureIds)

#Returns plate boundary, subduction zone and oceanic ridge data
def getPlateBoundaryData(time, topologyFeatures, rotationModel, earthRadius):
    
    #Initiate lists to be returned
    subZoneXYZ, overridingPlateIds, subductingPlateIds = [], [], []
    boundXYZ, boundPlateIds = [], []
    ridgeXYZ, ridgeIds = [], []
    boundaryType, shareBoundID = [], []
    
    #Get pygplates to resolve data at specified time
    resolvedTopologies, sharedBoundarySections = [], []
    pygplates.resolve_topologies(topologyFeatures, rotationModel, resolvedTopologies, int(time), sharedBoundarySections)
    
    #Loop through shared boundary sections and subsections
    for i, shareBound in enumerate(sharedBoundarySections):
        boundType = shareBound.get_feature().get_feature_type()
        isSubduction = boundType == pygplates.FeatureType.gpml_subduction_zone
        isOceanicRidge = boundType == pygplates.FeatureType.gpml_mid_ocean_ridge
        bType = 0
        if isOceanicRidge:
            bType = 1
        if isSubduction:
            bType = 2
        
        for sharedSubSection in shareBound.get_shared_sub_segments():
            
            #Get XYZ coordinates of shared subsections
            latLon = sharedSubSection.get_resolved_geometry().to_lat_lon_array()
            lon, lat = latLon[:, 1], latLon[:, 0]
            XYZ = lonLatToCartesian(lon, lat)
            
            #Append data for general plate boundaries
            sharedPlateIds = [i.get_resolved_feature().get_reconstruction_plate_id() for i in sharedSubSection.get_sharing_resolved_topologies()]
            for xyz in XYZ:
                boundXYZ.append(xyz)
                boundPlateIds.append(sharedPlateIds)
                boundaryType.append(bType)
                shareBoundID.append(i)
                
                #Append data for oceanic ridges
                if isOceanicRidge:
                    ridgeXYZ.append(xyz)
                    ridgeIds.append(sharedPlateIds)
            
            #Append data for subduction zones
            overAndSubPlates = sharedSubSection.get_overriding_and_subducting_plates(True)
            if isSubduction and (overAndSubPlates != None):
                overridingPlate, subductingPlate, subduction_polarity = overAndSubPlates
                overridingPlateId = overridingPlate.get_feature().get_reconstruction_plate_id()
                subductingPlateId = subductingPlate.get_feature().get_reconstruction_plate_id()
                for xyz in XYZ:
                    subZoneXYZ.append(xyz)
                    overridingPlateIds.append(overridingPlateId)
                    subductingPlateIds.append(subductingPlateId)
    
    #Convert data to numpy arrays and return data
    boundXYZ = np.array(boundXYZ) * earthRadius
    subZoneXYZ = np.array(subZoneXYZ) * earthRadius
    ridgeXYZ = np.array(ridgeXYZ) * earthRadius
    overridingPlateIds = np.array(overridingPlateIds)
    subductingPlateIds = np.array(subductingPlateIds)
    boundaryType = np.array(boundaryType)
    shareBoundID = np.array(shareBoundID)
    return (boundXYZ, subZoneXYZ, ridgeXYZ, boundPlateIds, overridingPlateIds, subductingPlateIds, ridgeIds, boundaryType, shareBoundID)

#Gets the reconstructed coastlines
def getCoastlineXYZ(time, rotationModel, coastLinesDirectory):
    reconstructedCoastlines = []
    Lat, Lon = [], []
    coastLines = pygplates.FeatureCollection(coastLinesDirectory)
    pygplates.reconstruct(coastLines, rotationModel, reconstructedCoastlines, time, group_with_feature=True)
    for feat in reconstructedCoastlines:
        for geom in feat[1]:
            for latLon in geom.get_reconstructed_geometry().to_lat_lon_array():
                Lat.append(latLon[0])
                Lon.append(latLon[1])
    return lonLatToCartesian(np.array(Lon), np.array(Lat))


#Returns a binary array with 1 representing a point as a continent and 0 as oceanic
#This particular algirithm is slow because the coastline data is way too high resolution
def calculateIsContinent(time, rotationModel, pointFeatures, coastLinesDirectory):
    reconstructedCoastlines = []
    coastLines = pygplates.FeatureCollection(coastLinesDirectory)
    pygplates.reconstruct(coastLines, rotationModel, reconstructedCoastlines, time, group_with_feature=True)
    
    #Loop through our point features and all coastline features
    isContinent = np.zeros(len(pointFeatures))
    for i, point in enumerate(pointFeatures):
        for feat in reconstructedCoastlines:
            coastlinePolyOnSphere = feat[1][0].get_reconstructed_geometry()
            
            #Check if point is on coastline feature
            min_distance_to_feature = pygplates.GeometryOnSphere.distance(
                point.get_geometry(),
                coastlinePolyOnSphere,
                geometry1_is_solid=True,
                geometry2_is_solid=True)
            if min_distance_to_feature == 0:
                isContinent[i] = 1
                break
    return isContinent

#Reads from data file or creates a new data file
def getIsContinent(time, rotationModel, pointFeatures, coastLinesDirectory):
    if not os.path.isdir('data'):
        os.mkdir('./data')
    
    #Read data file if it already exists, otherwise generate data and save as text file
    fileName = './data/time{}size{}.txt'.format(time, len(pointFeatures))
    if os.path.exists(fileName):
        fileLines = open(fileName).read().split('\n')[:-1]
        data = np.array(fileLines).astype(float)
    else:
        print('Generating new isContinent array')
        data = calculateIsContinent(time, rotationModel, pointFeatures, coastLinesDirectory)
        with open(fileName, 'w') as file:
            for i in data:
                file.write('{}\n'.format(i))
    return data.astype(bool)

    

#==============================================Speeds ========================================================================
#Given a boundary point, we get the speed of divergence or convergence
def getSpeed(xyz, rotationModel, timeFrom, timeTo, firstPlateId, secondPlateId):
    relativeRotation = rotationModel.get_rotation(int(timeTo), int(firstPlateId), int(timeFrom), int(secondPlateId))
    relativeRotation = relativeRotation.get_euler_pole_and_angle()
    axis = relativeRotation[0].to_xyz()
    angle = relativeRotation[1]
    rotationScipy = R.from_quat(quaternion(axis, angle))
    newXyz = rotationScipy.apply(xyz)
    return np.sum((newXyz - xyz)**2)**0.5 / ((timeFrom - timeTo)*2)

#Get speeds to be used in our speed transfer function
#boundXYZ, subZoneXYZ, oceanicRidgeXYZ, boundPlateIds, overridingPlateIds, subductingPlateIds, ridgePlateIds
#getSubductingSpeed(sphereXYZ, boundaryData, rotationModel, time-deltaTime, time)
def getSubductingSpeed(sphereXYZ, boundaryData, rotationModel, timeFrom, timeTo, numberToAverageOver, distIds):
    subZoneXYZ, overPlateIds, subPlateIds = boundaryData[1], boundaryData[4], boundaryData[5]
    speedsOfCollision = [getSpeed(xyz, rotationModel, timeFrom, timeTo, overPlateIds[i], subPlateIds[i]) for i, xyz in enumerate(subZoneXYZ)]
    speedsOfCollision = np.array(speedsOfCollision)
    speeds = speedsOfCollision[distIds]
    if len(speeds.shape) > 1:
        speeds = np.sum(speeds, axis=1) / speeds.shape[1]
    return normalizeArray(speeds), speedsOfCollision

#To calculate the ocean floor, we need the speed that plates are diverging at
#(oceanicRidgeXYZ, timeFrom, timeTo, rotationModel, sharedPlateIds)
def getRidgeSpreadingSpeeds(data, props, time, rotationModel):
    timeFrom, timeTo = time+props['deltaTime'], time
    spreadingSpeed = []
    for i, xyz in enumerate(data['RidgeXYZ']):
        if len(data['RidgeIds'][i]) == 2:
            spreadingSpeed.append(getSpeed(xyz, rotationModel, timeFrom, timeTo, data['RidgeIds'][i][0], data['RidgeIds'][i][1]))
        else:
            spreadingSpeed.append(0.0)
    return np.array(spreadingSpeed)
    
#======================================Code for Oceanic Ridge ===========================================================================
#Applies ocean depth based on equation from patrice's email
def applyOceanDepth(sphereXYZ, oceanicRidgeXYZ, heightMap, isOcean, spreadingSpeed, props):
    maxDepth = props['oceanDepth']
    ageProportion = props['ageProportion']
    ridgeDepth = props['ridgeDepth']
    
    distToRidge, closeRidgeID = KDTree(oceanicRidgeXYZ).query(sphereXYZ[isOcean])
    oceanFloorAge = np.abs(distToRidge) / (spreadingSpeed[closeRidgeID])
    oceanDepth = - (ridgeDepth + ageProportion * oceanFloorAge**0.5)
    oceanDepth[oceanDepth < -maxDepth] = -maxDepth
    heightMap[isOcean] = oceanDepth
    return heightMap

#======================================Code for Subduction ===========================================================================

#The height transfer function is given by the sigmoid function
def sigmoid(x, props):#pointToPassThrough=(0, 0.01), centre=2.0):
    pointToPassThrough, centre = props['pointToPassThrough'], props['centre']
    xPass, yPass = pointToPassThrough[0], pointToPassThrough[1]
    spread = (centre - xPass) / np.log((1/yPass) - 1)
    return 1 / (1 + np.exp((centre - x) / spread))

#The distance transfer defines how much uplift we should apply to a point based on the points distance from a subdcution zone
#To Do: I can greatly simplify this using interpolation
def distanceTransfer(x, slopeUp=2.0, slopeDown=0.5, maxRange=1.0, maxHeight=1.0):
    m1 = np.abs(slopeUp)
    m2 =  - np.abs(slopeDown)
    t = 0.5 * (maxRange + maxHeight * ((1/m1) + (1/m2)))
    y = np.zeros(x.shape)
    
    trapezoidalCase = m1 * t > maxHeight
    triangleCase = 1 - trapezoidalCase
        
    #The first line
    y1 = x * m1
    y1 *= (x < maxHeight / m1)

    #The second line
    plateauLength = (m1 * t - maxHeight) * 2
    y2 = np.ones(x.shape) * maxHeight
    y2 *= (x >= maxHeight / m1) * (x < (maxHeight + plateauLength) / m1)

    #The third line
    y3 = x * m2 + maxHeight - m2 * (maxHeight + plateauLength) / m1
    y3 *= (x >= (maxHeight + plateauLength) / m1)

    #Add the lines together and set negative values to zero
    y = y1 + y2 + y3
    y *= (y >= 0) * trapezoidalCase
    
    #The triangle case
    #else:
    t = (- m2 * maxRange) / (m1 - m2)
    
    #The first line
    y1 = x * m1
    y1 *= (x < t)
    
    #The second line
    y2 = (m2 * x + t * (m1 - m2))
    y2 *= (x >= t)
    
    #Add the lines together and set negative values to zero
    yTriangle = y1 + y2
    yTriangle *= (yTriangle >= 0) * triangleCase
    return y + yTriangle

#For each point on the sphere, we get the distance to the subzones out of those that are still on the same plate
def getDistancesFromSubZones(sphereXYZ, subZoneXYZ, plateIds, overridingPlateIds):
    distances = np.zeros(len(sphereXYZ))
    for idx in np.unique(plateIds):
        subZonesOnPlate = subZoneXYZ[overridingPlateIds == idx]
        if not (subZonesOnPlate.size == 0):
            subKDTree = KDTree(subZonesOnPlate)
            distances[plateIds == idx] = subKDTree.query(sphereXYZ[plateIds == idx])[0]
    return distances


#Function for showing alternative plots
def showAlternativePlots(props, showUpliftTemplate, showContShelfTemplate, showHeightTransfer, showMeltingProfile):
    exit=False
    if showHeightTransfer:
        x = np.arange(-oceanDepth, maxMountainHeight, (maxMountainHeight+oceanDepth)/1000)
        y = sigmoid(x, pointToPassThrough=pointToPass, centre=cent)
        plt.plot(x, y)
        plt.show()
        exit = True
    
    if showUpliftTemplate:
        x = np.arange(0, 1, 0.01)
        y = props['upliftTemplate'](x)
        plt.plot(x, y)
        plt.show()
        exit = True
    
    if showContShelfTemplate:
        x = np.arange(0, 1, 0.01)
        y = props['continentalShelfTemplate'](x)
        plt.plot(x, y)
        plt.show()
        exit = True
    '''
    if showMeltingTemplate:
        x = np.arange(0, 1, 0.01)
        y = meltingTemplate(x)
        plt.plot(x, y)
        plt.show()
        exit = True
    '''
    
    if showMeltingProfile:
        x = np.arange(props['weightInfluenceRange'])
        y = 1 / (np.exp((x - props['meltingSigmoidCentre']) * props['meltingSigmoidSteepness']) + 1)
        plt.plot(x, y)
        plt.show()
        exit = True
    
    if exit:
        sys.exit("We don't run the main code when showing alternative plots")

def initializeEarth(sphereXYZ, rotationModel, data, props):
    coastLinesDirectory = props['coastLinesDirectory']
    oceanShelfLength = props['oceanShelfLength']
    oceanDepth = props['oceanDepth']
    timeFrom = props['timeFrom']
    
    #Create initial heightMap of zeros and get other data we need for initializing earth
    heightMap = np.zeros(len(sphereXYZ))
    sphereLon, sphereLat = cartesianToLonLat(sphereXYZ[:, 0], sphereXYZ[:, 1], sphereXYZ[:, 2])
    pointFeatures = createPointFeatures(sphereLon, sphereLat)
    isContinent = getIsContinent(timeFrom, rotationModel, pointFeatures, coastLinesDirectory)
    coastXYZ = getCoastlineXYZ(timeFrom, rotationModel, coastLinesDirectory)
    isOcean = (1 - isContinent).astype(bool)
    coastXYZ *= props['earthRadius']
    heightMap[isContinent] = 0.01
       
    #Apply ocean floors
    spreadingSpeed = getRidgeSpreadingSpeeds(data, props, timeFrom, rotationModel) 
    spreadingSpeed[spreadingSpeed == 0] = 0.0001 #Avoid division by zero
    heightMap = applyOceanDepth(sphereXYZ, data['RidgeXYZ'], heightMap, isOcean, spreadingSpeed, props)
    
    #Apply mountain heights nearby subduction regions
    distToSubZone, closeSubzoneID = KDTree(data['SubZOneXYZ']).query(sphereXYZ[isContinent])
    subSpeed = getSubductingSpeed(sphereXYZ, data['BoundaryData'], rotationModel, timeFrom+1, timeFrom, props['numberToAverageOver'], closeSubzoneID)[0]
    heightMap[isContinent] += props['upliftTemplate'](distToSubZone / props['maxUpliftRange']) * props['maxMountainHeight'] * subSpeed**0.3
    
    #Apply oceanic shelves nearby coastline boundaries
    distToCoast, closeCoastID = KDTree(coastXYZ).query(sphereXYZ[isOcean])
    isCloseEnough = distToCoast < oceanShelfLength
    heightMap[isOcean] = props['continentalShelfTemplate'](distToCoast / oceanShelfLength) * oceanDepth * (isCloseEnough) + heightMap[isOcean] * (1 - isCloseEnough)
    return heightMap


#Anything that we wish to run over multiple CPUs, should be called here

#Anything that we can calculate before running the main simulation should be called here
#Multiple CPUs will run this function in parallel, but the CPUs can not communicate with each other
#We also can't pass too complicated objects to this function
def functionToParralelize(iteration, sphereXYZ, props, time):
    platePolygonsDirectory = props['platePolygonsDirectory']
    numberToAverageOver = props['numberToAverageOver']
    rotFileLoc = props['rotationsDirectory']
    deltaTime = props['deltaTime']
    if time > 250:
        platePolygonsDirectory = props['platePolygonsDirectory400MYA']
    
    #Create pygplates objects
    rotationModel = pygplates.RotationModel(rotFileLoc)
    sphereLon, sphereLat = cartesianToLonLat(sphereXYZ[:, 0], sphereXYZ[:, 1], sphereXYZ[:, 2])
    pointFeatures = createPointFeatures(sphereLon, sphereLat)
    
    #General data obtained from pygplates
    plateIds = getPlateIdsAtTime(time, rotationModel, pointFeatures, platePolygonsDirectory) 
    boundaryData = getPlateBoundaryData(time, platePolygonsDirectory, rotationModel, props['earthRadius'])
    
    #Data about subduction and oceanic ridges
    distIds = KDTree(boundaryData[1]).query(sphereXYZ, k=numberToAverageOver)[1]
    subSpeedTransfer, speedsOfCollision = getSubductingSpeed(sphereXYZ, boundaryData, rotationModel, time-deltaTime, time, numberToAverageOver, distIds)
    distToSub = getDistancesFromSubZones(sphereXYZ, boundaryData[1], plateIds, boundaryData[4])
    isOveriding = np.isin(plateIds, np.unique(boundaryData[4]))
    
    ridgeToVertexID = KDTree(sphereXYZ).query(boundaryData[2], k=props['resolution'] // props['divergingContinentLoweringRange'])[1]
    
    #distToRidge, distToRidgeID = KDTree(boundaryData[2]).query(sphereXYZ)
    #ridgeToVertexID = distToRidgeID[distToRidge <= props['divergingContinentLoweringRange']]
    
    #Used for moving plates and remeshing sphere
    movedSphereXYZ = movePlates(rotationModel, sphereXYZ, plateIds, time, deltaTime)
    distToSubZoneFromMovedSphere = KDTree(boundaryData[0]).query(movedSphereXYZ)[0]
    return (iteration, boundaryData, plateIds, subSpeedTransfer, speedsOfCollision, distToSub, 
            isOveriding, ridgeToVertexID, movedSphereXYZ, distToSubZoneFromMovedSphere)

#Move tectonic plates along the sphere
def movePlates(rotationModel, sphereXYZ, plateIds, time, deltaTime):
    newSphere = np.copy(sphereXYZ)
    for idx in np.unique(plateIds):
        stageRotation = rotationModel.get_rotation(int(time-deltaTime), int(idx), int(time)).get_euler_pole_and_angle()
        axisLatLon = stageRotation[0].to_lat_lon()
        axis = lonLatToCartesian(axisLatLon[1], axisLatLon[0])
        angle = stageRotation[1]
        rotationScipy = R.from_quat(quaternion(axis, angle))
        newSphere[plateIds == idx] = rotationScipy.apply(newSphere[plateIds == idx])
    return newSphere

#We use the griddata interpolation from the scipy library to remesh our sphere whilst maintianing heights
#def movePlatesAndRemeshSphere(sphereXYZ, heightMap, plateIds, time, deltaTime,  rotationModel, plateBoundsXYZ, boundIds):
def movePlatesAndRemeshSphere(sphereXYZ, heightMap, time, rotationModel, data, props):
    distToSubZone = data['distToSubZoneFromMovedSphere']
    movedSphereXYZ = data['movedSphereXYZ']
    deltaTime = props['deltaTime']
    plateIds = data['PlateIds']
    numberOfCPUS = mp.cpu_count()
    
    #Get longitude and latitude data
    sphereLonLat = np.stack(cartesianToLonLat(sphereXYZ[:, 0], sphereXYZ[:, 1], sphereXYZ[:, 2]), axis=1)
    movedLonLat = np.stack(cartesianToLonLat(movedSphereXYZ[:, 0], movedSphereXYZ[:, 1], movedSphereXYZ[:, 2]), axis=1)
    
    #When two plates collide into each other, some vertices of the two plates will overlap
    #In the case of subduction, the vertices of the mountains will be higher than those of the subducting plate
    #We want our interpolation algorithm to sample from the mountains, not the subducting plate
    #To do so, we use a clustering algorithm (DBSCAN) to check if vertices are closer than usual, 
    #and filter out vertices that are too far away from plate boundaries 
    
    #Use clustering algorithm to find vertices that are too close
    spacing = (150 / props['resolution'])
    lonLatForClustering = np.copy(movedLonLat)
    lonLatForClustering[:, 0] /= 2
    cluster = DBSCAN(eps=spacing, min_samples=2, n_jobs=numberOfCPUS).fit(lonLatForClustering)
    isCluster = (cluster.labels_ != -1)
    
    #Filter out vertices that are too far away from plate boundaries
    isCloseEnough = (distToSubZone < 300 * deltaTime * (np.cos(np.radians(movedLonLat[:, 1])) + 0.1))
    isOverlapping = isCloseEnough * isCluster
    
    #We set overlapping vertices to the height of its 6 nearest neighbours
    #So subducting vertices are now at mountain heights.
    spherePointsInCluster = movedSphereXYZ[isOverlapping]
    dists, ids = KDTree(movedSphereXYZ).query(spherePointsInCluster, k=6, workers=numberOfCPUS)
    neighbourHeights = heightMap[ids]
    maxNeighbourHeights = np.max(neighbourHeights, axis=1)
    heightMap[isOverlapping] = maxNeighbourHeights
    
    #The first griddata interpolation gives better results, but leaves NaNs behind
    #The second griddata interpolation is used to fill in the NaNs but with a slightly less accurate interpolation
    interpolatedHeights = griddata(movedLonLat, heightMap, sphereLonLat)
    whereNAN = np.argwhere(np.isnan(interpolatedHeights))
    interpolatedHeights[whereNAN] = griddata(movedLonLat, heightMap, sphereLonLat[whereNAN], method='nearest')
    return interpolatedHeights

#When mountains become too high, we reduce height and increase breadth of the mountain
def meltMountains(sphereXYZ, heightMap, subSpeedTransfer, props):
    maxMeltingDistance = props['maxMeltingDistance']
    baseMeltingUplift = props['baseMeltingUplift']
    heightThreshold = props['heightThreshold']
    gravityStrength = props['gravityStrength']
    
    #Get points on mountains that are too high
    isTooHigh = (heightMap > heightThreshold)
    tooHighXYZ = sphereXYZ[isTooHigh]
    tooHighHeights = heightMap[isTooHigh]
    
    #Stop runnning this function if there are no vertices above the height threshold
    if len(tooHighXYZ) == 0:
        return heightMap
    
    #Find clusters of too high vertices which will be considered as one mountain range each
    spacing = (2 * 2 * np.pi * props['earthRadius'] / props['resolution'])
    cluster = DBSCAN(eps=spacing, min_samples=1).fit(tooHighXYZ)
    clusterIds = cluster.labels_
    
    #For each cluster, we find the volume of moutain above threshold
    for idx in np.unique(clusterIds):
        volumeAboveThreshold = np.sum(tooHighHeights[clusterIds == idx] - heightThreshold)
        
        #Then we find surounding points on mountain foot
        distToMountain, distToMountIds = KDTree(tooHighXYZ[clusterIds == idx]).query(sphereXYZ)
        isInMeltingRange = ((distToMountain < maxMeltingDistance) * (distToMountain != 0)).astype(bool)
        
        #Then we increase the heights of points at the foot of the mountain
        numOfPointsInCluster = len(clusterIds[clusterIds == idx])
        meltingUplift = meltingTemplate(distToMountain[isInMeltingRange] / maxMeltingDistance)
        meltingUplift = baseMeltingUplift * meltingUplift * volumeAboveThreshold / numOfPointsInCluster
        heightMap[isInMeltingRange] += meltingUplift * subSpeedTransfer[isInMeltingRange]
    
    #Points that are above the height threshold are then decreased by a bit
    heightMap -= gravityStrength * (heightMap > heightThreshold) * (heightMap - heightThreshold)**2
    return heightMap


#Returns the displacement of height due to melting mountains
def getMeltingUplift(XYZ, hMap, dists, props):
    weightRange = props['weightInfluenceRange']
    sigmoidCent = props['meltingSigmoidCentre']
    sigmoidSteep = props['meltingSigmoidSteepness']
    
    hMap = normalizeArray(hMap)
    r, theta, phi = cartesianToPolarCoords(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2])
    integrationConstant = np.sin(phi)
    upliftField = np.zeros(len(XYZ))
    for i in range(len(XYZ)):
        dist = dists[i]
        isCloseEnough = ((dist <= weightRange) * (dist != 0)).astype(bool)
        nearbyAverage = np.mean(hMap[isCloseEnough])
        numerator = (nearbyAverage - hMap[isCloseEnough]) * integrationConstant[isCloseEnough]
        denominator = np.exp((dist[isCloseEnough] - sigmoidCent) * sigmoidSteep) + 1
        upliftField[isCloseEnough] += numerator / denominator
    return upliftField


    
#Apply ocean floors based on equation from patrices email
def applyOceanFloors(sphereXYZ, heightMap, data, props, time, rotationModel):
    isOceanThreshold = props['isOceanThreshold']
    oceanShelfLength = props['oceanShelfLength']
    ageProportion = props['ageProportion']
    ridgeDepth = props['ridgeDepth']
    maxDepth = props['oceanDepth']
    
    #Find the speed at which ridges are spreading
    spreadingSpeed = getRidgeSpreadingSpeeds(data, props, time, rotationModel)
    spreadingSpeed[spreadingSpeed == 0] = 0.0001
    
    #Identify land and ocean
    isOcean = (heightMap < isOceanThreshold)
    isLand = (1-isOcean).astype(bool)
    
    #Only apply ocean floors to vertices that are far away enough from land
    distToLand, distToLandID = KDTree(sphereXYZ[isLand]).query(sphereXYZ[isOcean])
    isFarFromLand = distToLand > oceanShelfLength
    isOcean[isOcean] = isFarFromLand
    
    #Apply ocean floors
    distToRidge, closeRidgeID = KDTree(data['RidgeXYZ']).query(sphereXYZ[isOcean])
    oceanFloorAge = np.abs(distToRidge) / (spreadingSpeed[closeRidgeID])
    oceanDepth = - (ridgeDepth + ageProportion * oceanFloorAge**0.5)
    oceanDepth[oceanDepth < -maxDepth] = heightMap[isOcean][oceanDepth < -maxDepth]
    heightMap[isOcean] = oceanDepth
    return heightMap

#Creates lines as pyvista mesh
def pyvistaLinesFromPoints(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly

#============================ Main Code =====================================================================================================
#Function for running the main simulation. It can be called from other python scripts to batch run simulations.
#All simulation properties are contained in the dictionary 'props'
def runMainTectonicSimulation(props):
    
    #Create the initial spherical mesh and rotation model
    earthRadius, resolution = props['earthRadius'], props['resolution']
    sphereMesh = pv.Sphere(radius=earthRadius, theta_resolution=resolution, phi_resolution=resolution)
    sphereXYZ, sphereFaces = sphereMesh.points.copy(), sphereMesh.faces.copy()
    sphereLonLat = np.stack(cartesianToLonLat(sphereXYZ[:, 0], sphereXYZ[:, 1], sphereXYZ[:, 2]), axis=1)
    rotationModel = pygplates.RotationModel(props['rotationsDirectory'])
    times = np.arange(props['timeFrom'], props['timeTo'], - props['deltaTime'])
    
    #We use parallel processing to pre-calculate slow parts of our code
    parallelOutput = []
    pool = mp.Pool(mp.cpu_count())
    for i, time in enumerate(times):
        parallelOutput.append(pool.apply_async(functionToParralelize, args=(i, sphereXYZ, props, time)))
    pool.close()
    pool.join()
    
    #Extract the output data from our parallel processing run
    #The list 'dataAtTimes' will contain a list of dictionaries containing all our precalculated data
    #Each dictionary contains data at a different time step of our simulation
    dataAtTimes = []
    parallelOutput = [i.get() for i in parallelOutput]
    parallelOutput.sort(key=lambda x: x[0]) 
    for i in parallelOutput:
        data = {}
        data['BoundaryData'] = i[1]
        data['PlateIds'] = i[2]
        data['SubductionSpeeds'] = i[3]
        data['SpeedOfCollision'] = i[4]
        data['DistToSubzones'] = i[5]
        data['IsOveriding'] = i[6]
        data['ridgeToVertexID'] = i[7]
        data['movedSphereXYZ'] = i[8]
        data['distToSubZoneFromMovedSphere'] = i[9]
        
        boundaryData = i[1]
        data['BoundaryXYZ'] = boundaryData[0]
        data['SubZOneXYZ'] = boundaryData[1]
        data['RidgeXYZ'] = boundaryData[2]
        data['BoundaryPlateIds'] = boundaryData[3]
        data['OverridingPlateIds'] = boundaryData[4]
        data['SubductingPlateIds'] = boundaryData[5]
        data['RidgeIds'] = boundaryData[6]
        data['boundaryType'] = boundaryData[7]
        data['shareBoundID'] = boundaryData[8]
        dataAtTimes.append(data)
    
    #Initialize earth
    data = dataAtTimes[0]
    heightMap = initializeEarth(sphereXYZ, rotationModel, data, props)
    
    
    #Create a sphere of lower resolution than the main sphere. 
    #The melting mountains algorithm will use this sphere's resolution.
    smallSphere = pv.Sphere(radius=earthRadius, theta_resolution=100, phi_resolution=100)
    smallXYZ = smallSphere.points
    sLon, sLat = cartesianToLonLat(smallXYZ[:, 0], smallXYZ[:, 1], smallXYZ[:, 2])
    sLonLat = np.stack((sLon, sLat), axis=1)
    distArray = distance.cdist(smallXYZ, smallXYZ, 'euclidean')
    
    #Create earth mesh for plotting
    heightMapForSphere = (props['heightAmplificationFactor'] * heightMap + props['earthRadius'])
    sphereXYZwithHeights = setRadialComponent(sphereXYZ, heightMapForSphere)
    sphereMesh = pv.PolyData(sphereXYZwithHeights, sphereFaces)
    
    #Create mesh showing boundary data
    subZoneMesh = pv.PolyData(data['SubZOneXYZ']*1.05)
    subZoneMesh.point_arrays['speed'] = data['SpeedOfCollision']
    subZoneMesh.point_arrays['orientation'] = data['SubZOneXYZ']
    subGlyph = subZoneMesh.glyph(geom=pv.Arrow(scale=10, direction=(-1.0, 0.0, 0.0)), scale='speed', orient="orientation")
    
    #Set up plotter
    if props['showMainPlot']:
        plotter = pv.Plotter()
        boundAct = plotter.add_mesh(data['BoundaryXYZ'])
        subZoneAct = plotter.add_mesh(subGlyph, color='b')
        ridgeAct = plotter.add_mesh(data['RidgeXYZ']*1.05, color='g')
        plotter.add_mesh(sphereMesh, scalars=heightMap, cmap='gist_earth')
        plotter.add_scalar_bar()
        plotter.add_axes(interactive=True)
        print('Please select viewing orientation and then press q to itterate over animation frames')
        plotter.show(auto_close=False, window_size=[800, 608])
        if props['saveAnimation']:
            plotter.open_movie('TectonicSimulation.mp4')
            plotter.write_frame()
    
    #Write data to file
    if props['writeDataToFile']:
        outputDirectory = createXYZOutputDirectory(props)
        writeXYZData(sphereXYZwithHeights, heightMap, outputDirectory, props['timeFrom'])
    
    #Run simulation through various times
    for i, time in enumerate(times[1:]):
        
        #Get precalculated data from parrallelization
        data = dataAtTimes[i+1]
        subSpeedTransfer = data['SubductionSpeeds']
        oceanicRidgeXYZ = data['RidgeXYZ']
        distToSub = data['DistToSubzones']
        isOveriding = data['IsOveriding']
        plateBounds = data['BoundaryXYZ']
        subZoneXYZ = data['SubZOneXYZ']
        
        #Apply subduction uplift
        heightTrans = sigmoid(heightMap, props)
        distanceTrans = props['upliftTemplate'](distToSub * subSpeedTransfer / props['maxUpliftRange']) * isOveriding
        uplift = heightTrans * distanceTrans * subSpeedTransfer
        heightMap += props['baseSubductionUplift'] * props['deltaTime'] * uplift
        
        
        
        
        
        
        
        
        '''
        #When continental plates diverge, we decrease the height at the continental ridge until it becomes an ocean
        ridgeToVertexID = data['ridgeToVertexID']
        vertexIsOnLand = (heightMap[ridgeToVertexID.flatten()] > props['isOceanThreshold'])
        heightMap[ridgeToVertexID.flatten()] -= props['loweringRate'] * props['deltaTime'] * vertexIsOnLand
        '''
        
        boundaryType = data['boundaryType']
        sharedBoundID = data['shareBoundID']
        boundaryIndex = np.arange(plateBounds.shape[0])
        
        lineCentres, boundLines, lineMesh = [], [], []
        for i in np.unique(sharedBoundID):
            sharesThis = (i==sharedBoundID)
            idx = boundaryIndex[sharesThis]
            bLines = np.array([idx[:-1], idx[1:]]).T.astype(int)
            if np.all(boundaryType[sharesThis] == 1):
                for line in bLines:
                    boundLines.append(line)
                    lineCentres.append(np.mean((plateBounds[line[0]], plateBounds[line[1]]), axis=0))
            
            #Add lines to pyvista plot
            #print(plateBounds.shape)
            #print(idx)
            #for line in pyvistaLinesFromPoints(plateBounds[idx]):
                #lineMesh.append(line)
        boundLines = np.array(boundLines)
        lineCentres = np.array(lineCentres)
        
        #Create a list of line segments represented by 2 xyz vertex coordinates
        distToLines, distToLinesIds = KDTree(lineCentres).query(sphereXYZ)
        lineSegmentXYZ = plateBounds[boundLines[distToLinesIds]]
        
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
        divergeLowering = 300 / (1 + np.exp(distToBound/60))
        heightMap -= divergeLowering * (heightMap > props['isOceanThreshold'])
        
        #ridgeToVertexID = data['ridgeToVertexID']
        #vertexIsOnLand = (heightMap[ridgeToVertexID.flatten()] > props['isOceanThreshold'])
        
        #heightMap[ridgeToVertexID.flatten()] -= props['loweringRate'] * props['deltaTime'] * vertexIsOnLand
        #print(heightMap.shape)
        #print(divergeLowering.shape)
        #print(vertexIsOnLand)
        #heightMap[vertexIsOnLand] -= divergeLowering[vertexIsOnLand]
            
        
        
        
        #Move plates and remesh using interpolation
        heightMap = movePlatesAndRemeshSphere(sphereXYZ, heightMap, time, rotationModel, data, props)
        heightMap = applyOceanFloors(sphereXYZ, heightMap, data, props, time, rotationModel)
        
        #Melting mountains algorithm
        sHeightMap = griddata(sphereLonLat, heightMap, sLonLat, method='nearest')
        upField = getMeltingUplift(smallXYZ, sHeightMap, distArray, props)
        upField = normalizeArray(upField, minValue=-1.0, maxValue=1.0)
        upFieldBig = griddata(sLonLat, sHeightMap, sphereLonLat, method='nearest')
        heightMap -= props['meltingRate'] * upFieldBig * props['deltaTime']
        
        heightMapForSphere = (props['heightAmplificationFactor'] * heightMap + props['earthRadius'])
        sphereXYZwithHeights = setRadialComponent(sphereXYZ, heightMapForSphere)
        
        #Update coordinates
        if props['showMainPlot']:
            plotter.update_coordinates(sphereXYZwithHeights, mesh=sphereMesh)
            plotter.update_scalars(heightMap, render=False, mesh=sphereMesh)
            
            #Remove old meshes since old meshes might have different number of vertices
            plotter.remove_actor(boundAct)
            plotter.remove_actor(subZoneAct)
            plotter.remove_actor(ridgeAct)
            
            #Add updated meshes
            boundAct = plotter.add_mesh(plateBounds)
            subZoneAct = plotter.add_mesh(subZoneXYZ*1.05, color='b')
            ridgeAct = plotter.add_mesh(oceanicRidgeXYZ*1.05, color='g')
            
            if not props['autoRunSimulation']:
                plotter.show(auto_close=False)
            if props['saveAnimation']:
                plotter.write_frame()
        
        if props['writeDataToFile']:
            writeXYZData(sphereXYZwithHeights, heightMap, outputDirectory, time)
        


        
#=============================Template Profile Curves ================================================================================
#
#To stop multiple CPUs from running our main code during parallelization, we check that the scope is __main__
if __name__ == '__main__':
    props = {}
    
    #We define a few template curves here
    #When manipulating these curves, makes sure to set 'showTemplate' variables to true 
    
    #Template curve for the distance transfer for subduction uplift
    upliftTemplatePoints = np.array([
        [-100, 0.0],
        [-1.0, 0.0],
        [-0.101, 0.0],
        [-0.1, 0.0],
        [0, 0.4],
        [0.19, 1.0],
        [0.21, 1.0],
        [0.5, 0.5],
        [0.99, 0.0],
        [1.0, 0.0],
        [100.0, 0.0]
        ])
    props['upliftTemplate'] = interp1d(upliftTemplatePoints[:, 0], upliftTemplatePoints[:, 1], kind='quadratic')
    showUpliftTemplate = False
    
    #Template curve for continental shelves
    continentalShelfPoints = np.array([
        [0, 0],
        [0.01, -0.04],
        [0.25, -0.1],
        [0.49, -0.196],
        [0.5, -0.2],
        [0.56, -1.0],
        [0.7, -4.5],
        [0.75, -4.66666666],
        [1.0, -5.5],
        [1.01, -5.5],
        [4.0, -5.5],
        [9.9, -5.5],
        [10, -5.5],
        [1000, -5.5]
        ])
    continentalShelfPoints[:, 1] /= 5.5
    props['continentalShelfTemplate'] = interp1d(continentalShelfPoints[:, 0], continentalShelfPoints[:, 1], kind='quadratic')
    showContShelfTemplate = False
    
    #============================================== Simulation Properties ==================================================
    #Directory for data files
    props['platePolygonsDirectory'] = './Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies_PMAG.gpmlz'
    props['platePolygonsDirectory400MYA'] = './Matthews_etal_GPC_2016_Paleozoic_PlateTopologies_PMAG.gpmlz'
    props['rotationsDirectory'] = './Matthews_etal_GPC_2016_410-0Ma_GK07_PMAG.rot'
    props['coastLinesDirectory'] = './Matthews_etal_GPC_2016_Coastlines.gpmlz'

    #Set the time range of simulation and time steps
    props['timeFrom'], props['timeTo'], props['deltaTime'] = 200, 0, 5

    #General earth properties
    props['resolution'] = 400 #Resolution of sphere
    props['earthRadius'] = 6371 #Radius of the earth in kilometres
    props['heightAmplificationFactor'] = 60 #For visualization, we amplify the height to make topography more visible

    #Properties for subduction
    props['maxUpliftRange'] = 700 #Maximum distance that our subduction algorithm has an effect on
    props['maxMountainHeight'] = 8 #Max mountain height (Only used in earth initialization)
    props['baseSubductionUplift'] = 0.3 #Defines how fast mountains grow due to subduction
    props['numberToAverageOver'] = 40 #The speed transfer takes the average collision speed of several nearby subduction regions
    props['pointToPassThrough'] = (0, 0.05) #Our height transfer is defined by a sigmoid that passes through this point
    props['centre'] = 3.0 #Our height transfer is a sigmoid with this as it's centre
    showHeightTransfer = False

    #Used for defining depth of oceanic ridges
    props['oceanDepth'] = 5.5
    props['ridgeDepth'] = 2.5 
    props['ageProportion'] = 0.35
    props['oceanShelfLength'] = 400 #The length of oceanic shelves
    props['isOceanThreshold'] = -1.0 #For the purpose of our algorithm, we identify oceans to be lower than 0, it gives better results

    #When continents diverge, we lower the land around the diverging boundary until it becomes an ocean
    props['divergingContinentLoweringRange'] = 80
    props['loweringRate'] = 0.4

    #Melting mountains refers to mountains collapsing under their own weight over time
    props['meltingRate'] = 0.005
    props['weightInfluenceRange'] = 1000
    props['meltingSigmoidCentre'] = 500
    props['meltingSigmoidSteepness'] = 1/140
    showMeltingProfile = False

    #Some control parameters for how we want the simulation to run
    props['autoRunSimulation'] = True #If false, the user must press 'q' to advance a step in the simulation
    props['saveAnimation'] = False #Do we want to save the simulation as an mp4 file?
    props['showMainPlot'] = True #If false, we don't show any plots during the simulation
    props['writeDataToFile'] = True #Do we want to save data as a csv file?
    
    #If any of the show templates booleans are true, then we show a plot of the template curves and don't run the main simulation
    showAlternativePlots(props, showUpliftTemplate, showContShelfTemplate, showHeightTransfer, showMeltingProfile)
    runMainTectonicSimulation(props)
        
        
    
    
    
    
    

