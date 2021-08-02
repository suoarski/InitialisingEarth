
import os
import pygplates
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from IPython.display import Video
from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata
from scipy.spatial.transform import Rotation as R

import time as tme
from PlateBoundaries import Boundaries
from EarthsAssistant import EarthAssist




#================================================== Earth Class ========================================================
class Earth:
    mainDirectory = os.path.dirname(os.path.dirname(os.path.abspath('')))
    
    def __init__(self,
                 startTime = 30,
                 endTime = 0,
                 deltaTime = 5,
                 earthRadius = 6378.137,
                 heightAmplificationFactor = 30,
                 platePolygonsDirectory = None,
                 rotationsDirectory = None,
                 initialElevationFilesDir = None,
                 plateIdsDirectory = None,
                 movieOutputDir = 'TectonicSimulation.mp4',
                 animationFramesPerIteration = 8,
                 numOfNeighbsForRemesh = 6,
                 clusterThresholdProportion = 300 / 360,
                 minClusterSize = 3,
                 moveTectonicPlates = True,
                 useKilometres = True,
                 useGospl = True,
                 
                 simulateSubduction = True,
                 baseUplift = 2,
                 distTransRange = 1000, 
                 numToAverageOver = 10
                ):
        
        #Set default directory locations if none other specified
        if platePolygonsDirectory == None:
            platePolygonsDirectory = Earth.mainDirectory + '/dataPygplates/Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies_PMAG.gpmlz'
        if rotationsDirectory == None:
            rotationsDirectory = Earth.mainDirectory + '/dataPygplates/Matthews_etal_GPC_2016_410-0Ma_GK07_PMAG.rot'
        if initialElevationFilesDir == None:
            initialElevationFilesDir = Earth.mainDirectory + '/PaleoDEMS'
        if plateIdsDirectory == None:
            plateIdsDirectory = Earth.mainDirectory + '/PlateIdData'
        
        #Set attribute from class initialization
        self.startTime = startTime
        self.endTime = endTime
        self.deltaTime = deltaTime
        self.earthRadius = earthRadius
        self.heightAmplificationFactor = heightAmplificationFactor
        self.platePolygonsDirectory = platePolygonsDirectory
        self.rotationsDirectory = rotationsDirectory
        self.initialElevationFilesDir = initialElevationFilesDir
        self.plateIdsDirectory = plateIdsDirectory
        self.movieOutputDir = movieOutputDir
        self.animationFramesPerIteration = animationFramesPerIteration
        self.numOfNeighbsForRemesh = numOfNeighbsForRemesh
        self.clusterThresholdProportion = clusterThresholdProportion
        self.minClusterSize = minClusterSize
        self.moveTectonicPlates = moveTectonicPlates
        self.useKilometres = useKilometres
        
        self.simulateSubduction = simulateSubduction
        self.baseUplift = baseUplift
        self.distTransRange = distTransRange
        self.numToAverageOver = numToAverageOver
        
        
        #Pre-calculate commonly used attributes
        initData = self.getInitialEarth(self.startTime, initialElevationFilesDir=initialElevationFilesDir)
        self.earthFaces = pv.PolyData(initData).delaunay_2d().faces
        if not useKilometres:
            initData[:, 2] *= 1000 #Use metres incase we don't use kilometres
        
        self.lon = initData[:, 0]
        self.lat = initData[:, 1]
        self.lonLat = np.stack((initData[:, 0], initData[:, 1]), axis=1)
        self.movedLonLat = self.lonLat
        self.sphereXYZ = EarthAssist.polarToCartesian(self.earthRadius, initData[:, 0], initData[:, 1])
        self.heightHistory = [initData[:, 2]]
        self.heights = self.heightHistory[-1]
        self.timeHistory = [startTime]
        self.simulationTimes = np.arange(self.startTime, self.endTime-self.deltaTime, -self.deltaTime)
        self.rotationModel = pygplates.RotationModel(self.rotationsDirectory)
        self.pointFeatures = self.createPointFeatures(initData[:, 0], initData[:, 1])
        self.thetaResolution = len(np.unique(initData[:, 0])) - 1
        self.phiResolution = len(np.unique(initData[:, 1])) - 1
        
        #Gospl related variables
        self.tectonicDisplacementHistory = []
        self.useGospl = useGospl
    
    #Run the simulation over all specified times
    def runTectonicSimulation(self):
        for time in self.simulationTimes[:-1]:
            print('Currently simulating at {} Millions years ago'.format(time), end='\r')
            self.doSimulationStep(time)
            
    #Run a single simulation step
    def doSimulationStep(self, time):
        self.heights = np.copy(self.heightHistory[-1])
        self.timeHistory.append(time - self.deltaTime)
        plateIds = self.getPlateIdsAtTime(time)
        rotations = self.getRotations(plateIds, time)
        if self.simulateSubduction:
            self.doSubductionUplift(time, plateIds, rotations)
        if self.movePlates:
            self.movePlatesAndRemesh(plateIds, rotations)
        if self.useGospl:
            self.createTectonicDisplacements()
        
    #Run algorithm for moving plates and remeshing the sphere
    def movePlatesAndRemesh(self, plateIds, rotations):
        movedEarthXYZ = self.movePlates(plateIds, rotations)
        movedLonLat = EarthAssist.cartesianToPolarCoords(movedEarthXYZ)
        self.movedLonLat = np.stack((movedLonLat[1], movedLonLat[2]), axis=1)
        heights = self.remeshSphere(self.movedLonLat)
        self.heightHistory.append(heights)
    
    
    
    
    def createTectonicDisplacements(self, maxTectonicDisp=0.18):
        earthBeforeXYZ = self.getEarthXYZ(amplifier=1, iteration=-2)
        
        #Get earth's XYZ after moving plates but before the remesh
        heightsAfter = self.heights
        radius = heightsAfter + self.earthRadius
        earthAfterXYZ = EarthAssist.polarToCartesian(radius, self.movedLonLat[:, 0], self.movedLonLat[:, 1])
        
        #Calculate the tectonic displacements, and set maxTectonicDisp
        tectonicDisp = (earthAfterXYZ - earthBeforeXYZ)
        tectonicDisp *= maxTectonicDisp / np.max(np.linalg.norm(tectonicDisp, axis=1))
        self.tectonicDisplacementHistory.append(tectonicDisp)
    
    
    
    
    #Run algorithm for subduction uplift
    def doSubductionUplift(self, time, plateIds, rotations):
        boundaries = Boundaries(time, self, plateIds, rotations,
            baseUplift = self.baseUplift,
            distTransRange = self.distTransRange, 
            numToAverageOver = self.numToAverageOver)
        uplifts = boundaries.getUplifts()
        self.heights += uplifts
    
    #Get XYZ coordinates of earth at specified iteration (the default iteration is the latest iteration)
    def getEarthXYZ(self, iteration=-1, amplifier=None):
        if amplifier == None:
            amplifier = self.heightAmplificationFactor
        lon, lat = self.lonLat[:, 0], self.lonLat[:, 1]
        exageratedRadius = self.heightHistory[iteration] * amplifier + self.earthRadius
        earthXYZ = EarthAssist.polarToCartesian(exageratedRadius, lon, lat)
        return earthXYZ
    
    #Create pyvista mesh object of earth for plotting
    def getEarthMesh(self, iteration=-1, amplifier=None):
        earthXYZ = self.getEarthXYZ(iteration=iteration, amplifier=amplifier)
        earthMesh = pv.PolyData(earthXYZ, self.earthFaces)
        earthMesh['heights'] = self.heightHistory[iteration]
        return earthMesh
    
    #Create a plot of earth suitable for jupyter notebook at specified iteration
    def showEarth(self, iteration=-1):
        earthMesh = self.getEarthMesh(iteration=iteration)
        plotter = pv.PlotterITK()
        plotter.add_mesh(earthMesh, scalars=self.heightHistory[iteration])
        plotter.show(window_size=[800, 400])
    
    #Create an animation of the earth which is saved as an mp4 file in the current directory
    def animate(self, lookAtLonLat=[0, 0], cameraZoom=1.4, framesPerIteration=None):
        earthXYZ = self.getEarthXYZ(iteration=0)
        earthMesh = pv.PolyData(earthXYZ, self.earthFaces)
        if framesPerIteration == None:
            framesPerIteration = self.animationFramesPerIteration
        
        #Set up plotter for animation
        plotter = pv.Plotter()
        plotter.add_mesh(earthMesh, scalars=self.heightHistory[0], cmap='gist_earth')
        plotter.camera_position = 'yz'
        plotter.camera.zoom(cameraZoom)
        plotter.camera.azimuth = 180 + lookAtLonLat[0]
        plotter.camera.elevation = lookAtLonLat[1]
        plotter.show(auto_close=False, window_size=[800, 608])
        plotter.open_movie(self.movieOutputDir)
        for i in range(framesPerIteration):
            plotter.write_frame()
        
        #Draw frames of simulation
        for i in range(len(self.heightHistory)-1):
            earthXYZ = self.getEarthXYZ(iteration=i+1)
            plotter.update_coordinates(earthXYZ, mesh=earthMesh)
            plotter.update_scalars(self.heightHistory[i+1], render=False, mesh=earthMesh)
            for i in range(framesPerIteration):
                plotter.write_frame()
        plotter.close()
    
    #=================================================== Move Tectonic Plates =================================================
    #Get stage rotation data from pygplates and return a scipy rotation
    def getRotations(self, plateIds, time):
        rotations = {}
        for idx in np.unique(plateIds):
            stageRotation = self.rotationModel.get_rotation(int(time-self.deltaTime), int(idx), int(time))
            stageRotation = stageRotation.get_euler_pole_and_angle()

            #Create rotation quaternion from axis and angle
            axisLatLon = stageRotation[0].to_lat_lon()
            axis = EarthAssist.polarToCartesian(1, axisLatLon[1], axisLatLon[0])
            angle = stageRotation[1]
            rotations[idx] = R.from_quat(EarthAssist.quaternion(axis, angle))
        return rotations

    #Move tectonic plates along the sphere by applying rotations to vertices with appropriate plate ids
    def movePlates(self, plateIds, rotations):
        newXYZ = np.copy(self.sphereXYZ)
        for idx in np.unique(plateIds):
            rot = rotations[idx]
            newXYZ[plateIds == idx] = rot.apply(newXYZ[plateIds == idx])
        return newXYZ
    
    def getHeightsForRemesh(self, movedLonLat):
        heights = self.heights
        
        #Create clyinder
        m = self.thetaResolution
        n = self.phiResolution
        h = np.max(movedLonLat[:, 1]) - np.min(movedLonLat[:, 1])
        cylinderRadius = m * h / (np.pi * n * 2)
        cylinderXYZ = EarthAssist.cylindricalToCartesian(cylinderRadius, movedLonLat[:, 0], movedLonLat[:, 1])

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
        start = tme.time()
        heightsForRemesh = self.getHeightsForRemesh(movedLonLat)
        newHeights = griddata(movedLonLat, heightsForRemesh, self.lonLat)
        whereNAN = np.argwhere(np.isnan(newHeights))
        newHeights[whereNAN] = griddata(movedLonLat, heightsForRemesh, self.lonLat[whereNAN], method='nearest')
        return newHeights
    
    #================================================= Coordinate Transformation Functions ======================================
    

    #========================================== Reading From Data Sources =============================================
    #Read initial landscape data at specified time from file which is in the form of (lon, lat, height)
    @staticmethod
    def getInitialEarth(time, initialElevationFilesDir=None):
        if initialElevationFilesDir == None:
            initialElevationFilesDir = Earth.mainDirectory + '/PaleoDEMS'
        
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
    @staticmethod
    def createPointFeatures(lon, lat):
        pointsOnSphere = [pygplates.PointOnSphere(float(lat[i]), float(lon[i])) for i in range(len(lon))]
        pointFeatures = []
        for point in pointsOnSphere:
            pointFeature = pygplates.Feature()
            pointFeature.set_geometry(point)
            pointFeatures.append(pointFeature)
        return pointFeatures

    #Returns a list of plate Ids for points on our sphere
    @staticmethod
    def createPlateIdsAtTime(time, pointFeatures, platePolygonsDirectory=None, rotationsDirectory=None):
        if platePolygonsDirectory == None:
            platePolygonsDirectory = Earth.mainDirectory + '/dataPygplates/Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies_PMAG.gpmlz'
        if rotationsDirectory == None:
            rotationsDirectory = Earth.mainDirectory + '/dataPygplates/Matthews_etal_GPC_2016_410-0Ma_GK07_PMAG.rot'
        
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
    
    #Either read plate ids from file, or call functions to create new plate ids
    def getPlateIdsAtTime(self, time): 
        plateIdsDirectory = self.plateIdsDirectory
        
        #Create data folder if it doesn't already exists
        if not os.path.isdir(plateIdsDirectory):
            os.mkdir('./' + plateIdsDirectory)
        
        #Create a file name, and read from file if it already exists
        fileName = '{}/time{}size{}.txt'.format(plateIdsDirectory, time, len(self.lon))
        if os.path.exists(fileName):
            plateIds = pd.read_csv(fileName, header=None)
            plateIds = np.array(plateIds)[:, 0]
        
        #Otherwise calculate plate ids and write to file
        else:
            #Calculate plate ids
            print('Creating new plate ID file')
            pointFeatures = self.createPointFeatures(self.lon, self.lat)
            plateIds = self.createPlateIdsAtTime(time, 
                            pointFeatures, 
                            platePolygonsDirectory = self.platePolygonsDirectory, 
                            rotationsDirectory = self.rotationsDirectory)
            
            #Write to file
            with open(fileName, 'w') as file:
                for idx in plateIds:
                    file.write('{}\n'.format(idx))
        return plateIds

