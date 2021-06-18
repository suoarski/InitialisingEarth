
import os
import pygplates
import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from IPython.display import Video
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
from scipy.spatial.transform import Rotation as R

#================================================= Coordinate Transformation Functions ======================================
#Coordinate transformation from spherical polar to cartesian
def polarToCartesian(radius, theta, phi, useLonLat=True):
    if useLonLat == True:
        theta, phi = np.radians(theta+180.), np.radians(90. - phi)
    X = radius * np.cos(theta) * np.sin(phi)
    Y = radius * np.sin(theta) * np.sin(phi)
    Z = radius * np.cos(phi)
    
    #Return data either as a list of XYZ coordinates or as a single XYZ coordinate
    if (type(X) == np.ndarray):
        return np.stack((X, Y, Z), axis=1)
    else:
        return np.array([X, Y, Z])

#Coordinate transformation from cartesian to polar
def cartesianToPolarCoords(XYZ, useLonLat=True):
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    R = (X**2 + Y**2 + Z**2)**0.5
    theta = np.arctan2(Y, X)
    phi = np.arccos(Z / R)
    
    #Return results either in spherical polar or leave it in radians
    if useLonLat == True:
        theta, phi = np.degrees(theta), np.degrees(phi)
        lon, lat = theta - 180, 90 - phi
        lon[lon < -180] = lon[lon < -180] + 360
        return R, lon, lat
    else:
        return R, theta, phi

#Coordinate transformation functions from cartesian to cylindrical polar coordinates
def cartesianToCylindrical(X, Y, Z):
    r = (X**2 + Y**2)**0.5
    theta = np.arctan2(Y, X)
    return np.stack((r, theta, Z), axis=1)

#Coordinate transformation functions from cylindrical polar coordinates to cartesian
def cylindricalToCartesian(r, theta, Z, useDegrees=True):
    if useDegrees == True:
        theta = np.radians(theta+180.)
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    return np.stack((X, Y, Z), axis=1)

#Returns a rotation quaternion
def quaternion(axis, angle):
    return [np.sin(angle/2) * axis[0], 
            np.sin(angle/2) * axis[1], 
            np.sin(angle/2) * axis[2], 
            np.cos(angle/2)]












#========================================== Reading From Data Sources =============================================
#Read initial landscape data at specified time from file which is in the form of (lon, lat, height)
def getInitialEarth(time, initialElevationFilesDir='./PaleoDEMS'):
    
    #Get path of initial landscape data file at specified time
    paleoDemsPath = Path(initialElevationFilesDir)
    initialLandscapePath = list(paleoDemsPath.glob('**/*%03.fMa.csv'%time))[0]
    
    #Read data and split by newline and commas to create numpy array of data
    initialLandscapeFileLines = open(initialLandscapePath).read().split('\n')[1:-1]
    initLandscapeData = [line.split(',') for line in initialLandscapeFileLines]
    initLandscapeData = np.array(initLandscapeData).astype(float)
    
    #Set heights from metres to kilometers and return data
    initLandscapeData[:, 2] /= 1000
    return initLandscapeData

#Creates a list of point features for each vertex on our sphere
def createPointFeatures(lon, lat):
    pointsOnSphere = [pygplates.PointOnSphere(float(lat[i]), float(lon[i])) for i in range(len(lon))]
    pointFeatures = []
    for point in pointsOnSphere:
        pointFeature = pygplates.Feature()
        pointFeature.set_geometry(point)
        pointFeatures.append(pointFeature)
    return pointFeatures

#Returns a list of plate Ids for points on our sphere
def getPlateIdsAtTime(time, pointFeatures, 
        platePolygonsDirectory = './dataPygplates/Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies_PMAG.gpmlz',
        rotationsDirectory = './dataPygplates/Matthews_etal_GPC_2016_410-0Ma_GK07_PMAG.rot'):
    
    assignedPointFeatures = pygplates.partition_into_plates(
        platePolygonsDirectory,
        rotationsDirectory,
        pointFeatures,
        reconstruction_time=float(time),
        properties_to_copy = [
            pygplates.PartitionProperty.reconstruction_plate_id,
            pygplates.PartitionProperty.valid_time_period])
    featureIds = [feat.get_reconstruction_plate_id() for feat in assignedPointFeatures]
    return np.array(featureIds)


#================================================== Earth Class ========================================================
class Earth:
    def __init__(self,
                 startTime = 10,
                 endTime = 0,
                 deltaTime = 5,
                 earthRadius = 6378.137,
                 heightAmplificationFactor = 60,
                 platePolygonsDirectory = './dataPygplates/Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies_PMAG.gpmlz',
                 rotationsDirectory = './dataPygplates/Matthews_etal_GPC_2016_410-0Ma_GK07_PMAG.rot',
                 initialElevationFilesDir = './PaleoDEMS',
                 movieOutputDir = 'TectonicSimulation.mp4',
                 animationFramesPerIteration = 8,
                 numOfNeighbsForRemesh = 6,
                 clusterThresholdProportion = 300 / 360,
                 minClusterSize = 3
                ):
        
        #Set attribute from class initialization
        self.startTime = startTime
        self.endTime = endTime
        self.deltaTime = deltaTime
        self.earthRadius = earthRadius
        self.heightAmplificationFactor = heightAmplificationFactor
        self.platePolygonsDirectory = platePolygonsDirectory
        self.rotationsDirectory = rotationsDirectory
        self.initialElevationFilesDir = initialElevationFilesDir
        self.movieOutputDir = movieOutputDir
        self.animationFramesPerIteration = animationFramesPerIteration
        self.numOfNeighbsForRemesh = numOfNeighbsForRemesh
        self.clusterThresholdProportion = clusterThresholdProportion
        self.minClusterSize = minClusterSize
        
        #Pre-calculate commonly used attributes
        initData = getInitialEarth(self.startTime, initialElevationFilesDir=initialElevationFilesDir)
        self.lonLat = np.stack((initData[:, 0], initData[:, 1]), axis=1)
        self.sphereXYZ = polarToCartesian(1, initData[:, 0], initData[:, 1])
        self.heightHistory = [initData[:, 2]]
        self.simulationTimes = np.arange(self.startTime, self.endTime-self.deltaTime, -self.deltaTime)
        self.rotationModel = pygplates.RotationModel(self.rotationsDirectory)
        self.pointFeatures = createPointFeatures(initData[:, 0], initData[:, 1])
        self.earthFaces = pv.PolyData(initData).delaunay_2d().faces
        self.thetaResolution = len(np.unique(initData[:, 0])) - 1
        self.phiResolution = len(np.unique(initData[:, 1])) - 1
        self.totalIterations = 1
    
    #Run the simulation at specified times and append results to heightHistory
    def runTectonicSimulation(self):
        for time in self.simulationTimes:
            self.totalIterations += 1
            plateIds = getPlateIdsAtTime(time, self.pointFeatures, platePolygonsDirectory=self.platePolygonsDirectory, rotationsDirectory=self.rotationsDirectory)
            movedEarthXYZ = self.movePlates(plateIds, time)
            movedLonLat = cartesianToPolarCoords(movedEarthXYZ)
            movedLonLat = np.stack((movedLonLat[1], movedLonLat[2]), axis=1)
            heights = self.remeshSphere(movedLonLat)
            self.heightHistory.append(heights)
    
    #Get XYZ coordinates of earth at specified iteration (the default iteration is the latest iteration)
    def getEarthXYZ(self, iteration=-1):
        amplifier = self.heightAmplificationFactor
        lon, lat = self.lonLat[:, 0], self.lonLat[:, 1]
        exageratedRadius = self.heightHistory[iteration] * amplifier + self.earthRadius
        earthXYZ = polarToCartesian(exageratedRadius, lon, lat)
        return earthXYZ
    
    #Create a plot of earth suitable for jupyter notebook at specified iteration
    def showEarth(self, iteration=-1):
        earthXYZ = self.getEarthXYZ(iteration=iteration)
        earthMesh = pv.PolyData(earthXYZ, self.earthFaces)
        plotter = pv.PlotterITK()
        plotter.add_mesh(earthMesh, scalars=self.heightHistory[iteration])
        plotter.show(window_size=[800, 400])
    
    #Create an animation of the earth which is saved as an mp4 file in the current directory
    def animate(self):
        earthXYZ = self.getEarthXYZ(iteration=0)
        earthMesh = pv.PolyData(earthXYZ, self.earthFaces)
        
        #Set up plotter for animation
        plotter = pv.Plotter()
        plotter.add_mesh(earthMesh, scalars=self.heightHistory[0], cmap='gist_earth')
        plotter.show(auto_close=False, window_size=[800, 608])
        plotter.open_movie(self.movieOutputDir)
        plotter.write_frame()
        
        #Draw frames of simulation
        for i in range(self.totalIterations-1):
            earthXYZ = self.getEarthXYZ(iteration=i+1)
            plotter.update_coordinates(earthXYZ, mesh=earthMesh)
            plotter.update_scalars(self.heightHistory[i+1], render=False, mesh=earthMesh)
            for i in range(self.animationFramesPerIteration):
                plotter.write_frame()
        plotter.close()
    
    #=================================================== Move Tectonic Plates =================================================
    #Get stage rotation data from pygplates and return a scipy rotation
    def getRotation(self, time, plateId):
        stageRotation = self.rotationModel.get_rotation(int(time-self.deltaTime), int(plateId), int(time))
        stageRotation = stageRotation.get_euler_pole_and_angle()

        #Create rotation quaternion from axis and angle
        axisLatLon = stageRotation[0].to_lat_lon()
        axis = polarToCartesian(1, axisLatLon[1], axisLatLon[0])
        angle = stageRotation[1]
        return R.from_quat(quaternion(axis, angle))

    #Move tectonic plates along the sphere by applying rotations to vertices with appropriate plate ids
    def movePlates(self, plateIds, time):
        newXYZ = np.copy(self.sphereXYZ)
        for idx in np.unique(plateIds):
            rot = self.getRotation(time, idx)
            newXYZ[plateIds == idx] = rot.apply(newXYZ[plateIds == idx])
        return newXYZ
    
    def getHeightsForRemesh(self, movedLonLat):
        heights = self.heightHistory[-1]
        
        #Create clyinder
        m = self.thetaResolution
        n = self.phiResolution
        h = np.max(movedLonLat[:, 1]) - np.min(movedLonLat[:, 1])
        cylinderRadius = m * h / (np.pi * n * 2)
        cylinderXYZ = cylindricalToCartesian(cylinderRadius, movedLonLat[:, 0], movedLonLat[:, 1])

        #Run the clustering algorithm
        threshHoldDist = self.clusterThresholdProportion * 360 / m
        cluster = DBSCAN(eps=threshHoldDist, min_samples=self.minClusterSize).fit(cylinderXYZ)
        isCluster = (cluster.labels_ != -1)
        
        #Create KDTree to find nearest neighbours of each point in cluster
        pointsInClusterLonLat = cylinderXYZ[isCluster]
        clusterKDTree = cKDTree(pointsInClusterLonLat).query(pointsInClusterLonLat, k=self.numOfNeighbsForRemesh+1)
        
        #Get heights of nearest neighbours
        heightsInCluster = heights[isCluster]
        clusterPointsNeighboursId = clusterKDTree[1]
        neighbourHeights = heightsInCluster[clusterPointsNeighboursId[:, 1:]]

        #For points in cluster, set new heights to the maximum height of nearest neighbours
        newHeights = np.copy(heights)
        newHeights[isCluster] = np.max(neighbourHeights, axis=1)
        return newHeights

    def remeshSphere(self, movedLonLat):
        heightsForRemesh = self.getHeightsForRemesh(movedLonLat)
        newHeights = griddata(movedLonLat, heightsForRemesh, self.lonLat)
        whereNAN = np.argwhere(np.isnan(newHeights))
        newHeights[whereNAN] = griddata(movedLonLat, heightsForRemesh, self.lonLat[whereNAN], method='nearest')
        return newHeights


