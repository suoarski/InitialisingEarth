
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
from GosplManager import GosplManager
from scipy.interpolate import griddata
from scipy.spatial.transform import Rotation as R

import time as tme
from PlateBoundaries import Boundaries
from EarthsAssistant import EarthAssist



#================================================== Earth Class ========================================================
class Earth:
    mainDirectory = os.path.dirname(os.path.dirname(os.path.abspath('')))
    
    #Initiate with various optional parameters to change
    def __init__(self,
                 startTime = 30,
                 endTime = 0,
                 deltaTime = 5,
                 earthRadius = 6378137.,
                 heightAmplificationFactor = 30,
                 animationFramesPerIteration = 8,
                 
                 #Parameters related to subduction uplift
                 baseUplift = 1000,
                 numToAverageOver = 10,
                 distTransRange = 1000000,
                 
                 #Parameters related to diverge lowering
                 baseLowering = 2000,
                 maxLoweringDistance = 200000,
                 minMaxLoweringHeights = 8000,
                 
                 #Parameters related to remeshing
                 minClusterSize = 3,
                 numOfNeighbsForRemesh = 6,
                 clusterThresholdProportion = 300 / 360,
                 
                 #If any of these parameters are None, we use their default directory loctations
                 movieOutputDir = None,
                 npzSaveDirectory = None,
                 plateIdsDirectory = None,
                 rotationsDirectory = None,
                 platePolygonsDirectory = None,
                 initialElevationFilesDir = None,
                 
                 #Specify which algorithms to use during a simulation run
                 useGospl = True,
                 useKilometres = False,
                 moveTectonicPlates = True,
                 simulateSubduction = True,
                 useDivergeLowering = True
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
        if npzSaveDirectory == None:
            npzSaveDirectory = '/TectonicEarthSaves'
        if movieOutputDir == None:
            movieOutputDir = 'TectonicSimulation.mp4'
        
        #Set basic attribute from class initialization
        self.startTime = startTime
        self.endTime = endTime
        self.deltaTime = deltaTime
        self.earthRadius = earthRadius
        self.heightAmplificationFactor = heightAmplificationFactor
        self.animationFramesPerIteration = animationFramesPerIteration
        
        #Set attributes for subduction uplift
        self.baseUplift = baseUplift
        self.distTransRange = distTransRange
        self.numToAverageOver = numToAverageOver
        
        #Attributes related to diverge lowering
        self.baseLowering = baseLowering
        self.maxLoweringDistance = maxLoweringDistance
        self.minMaxLoweringHeights = minMaxLoweringHeights
        
        #Attributes related to remeshing
        self.clusterThresholdProportion = clusterThresholdProportion
        self.numOfNeighbsForRemesh = numOfNeighbsForRemesh
        self.minClusterSize = minClusterSize
        
        #Set directory related attributes
        self.initialElevationFilesDir = initialElevationFilesDir
        self.platePolygonsDirectory = platePolygonsDirectory
        self.rotationsDirectory = rotationsDirectory
        self.plateIdsDirectory = plateIdsDirectory
        self.npzSaveDirectory = npzSaveDirectory
        self.movieOutputDir = movieOutputDir
        
        #Specify which algorithm to run during the simulation
        self.useGospl = useGospl
        self.useKilometres = useKilometres
        self.useDivergeLowering = useDivergeLowering
        self.moveTectonicPlates = moveTectonicPlates
        self.simulateSubduction = simulateSubduction
        
        #Get initial topological data and convert to from kilometres to metres if chosen so
        initData = self.getInitialEarth(self.startTime, initialElevationFilesDir=initialElevationFilesDir)
        self.earthFaces = pv.PolyData(initData).delaunay_2d().faces
        if not useKilometres:
            initData[:, 2] *= 1000
        
        #Initiate coordinate related data
        self.lon = initData[:, 0]
        self.lat = initData[:, 1]
        self.lonLat = np.stack((initData[:, 0], initData[:, 1]), axis=1)
        self.sphereXYZ = EarthAssist.polarToCartesian(self.earthRadius, initData[:, 0], initData[:, 1])
        self.pointFeatures = self.createPointFeatures(initData[:, 0], initData[:, 1])
        self.thetaResolution = len(np.unique(initData[:, 0])) - 1
        self.phiResolution = len(np.unique(initData[:, 1])) - 1
        self.movedLonLat = self.lonLat
        
        #Random other variables
        self.timeHistory = [startTime]
        self.heightHistory = [initData[:, 2]]
        self.heights = self.heightHistory[-1]
        self.tectonicDispHistory = []
        self.rotationModel = pygplates.RotationModel(self.rotationsDirectory)
        self.simulationTimes = np.arange(self.startTime, self.endTime-self.deltaTime, -self.deltaTime)
        self.setPlateData(self.startTime)
        self.oceanFloorAge = self.boundaries.getOceanFloorAge()
    
    #Run the simulation over all specified times
    def runTectonicSimulation(self):
        for time in self.simulationTimes[:-1]:
            print('Currently simulating at {} Millions years ago'.format(time), end='\r')
            self.doSimulationStep(time)
            
    #Run a single simulation step
    def doSimulationStep(self, time):
        self.heights = np.copy(self.heightHistory[-1])
        self.timeHistory.append(time - self.deltaTime)
        self.setPlateData(time)
        if self.simulateSubduction:
            self.heights += self.boundaries.getUplifts()
        if self.useDivergeLowering:
            self.heights += self.boundaries.getDivergeLowering()
        if self.movePlates:
            self.movePlatesAndRemesh()
        if self.useGospl:
            self.createTectonicDisplacements()
    
    #Set current plateIds, rotations and plate boundaries
    def setPlateData(self, time):
        self.plateIds = self.getPlateIdsAtTime(time)
        self.rotations = self.getRotations(self.plateIds, time)
        self.boundaries = Boundaries(time, self, self.plateIds, self.rotations,
                                    baseUplift = self.baseUplift,
                                    distTransRange = self.distTransRange, 
                                    numToAverageOver = self.numToAverageOver,
                                    baseLowering = self.baseLowering,
                                    maxLoweringDistance = self.maxLoweringDistance,
                                    minMaxLoweringHeights = self.minMaxLoweringHeights
                                    )
    
    #================================================== Visualizations ========================================================
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
    
    #For all you flat earth believers out there, this function is dedicated to you!!!
    def getFlatEarthXYZ(self, iteration=-1, amplifier=None):
        if amplifier == None:
            amplifier = self.heightAmplificationFactor
        lon, lat = self.lonLat[:, 0], self.lonLat[:, 1]
        exageratedRadius = self.heightHistory[iteration] * amplifier / 100000 + self.earthRadius
        earthXYZ = np.stack((lon, lat, exageratedRadius)).T
        return earthXYZ
    
    #For all you flat earth believers out there, this function is dedicated to you!!! 
    def getFlatEarthMesh(self, iteration=-1, amplifier=None):
        earthXYZ = self.getFlatEarthXYZ(iteration=iteration, amplifier=amplifier)
        earthMesh = pv.PolyData(earthXYZ, self.earthFaces)
        earthMesh['heights'] = self.heightHistory[iteration]
        return earthMesh
    
    #Create a plot of earth suitable for jupyter notebook at specified iteration
    def showEarth(self, iteration=-1, showBounds=False):
        earthMesh = self.getEarthMesh(iteration=iteration)  
        plotter = pv.PlotterITK()
        plotter.add_mesh(earthMesh, scalars='heights')
        if showBounds:
            plotter.add_mesh(self.boundaries.getBoundaryMesh())
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
    
    #Save results of this simulation as an NPZ file
    def saveDataAsNPZ(self):
        if not os.path.isdir(self.npzSaveDirectory):
            os.mkdir('./{}'.format(self.npzSaveDirectory))
        
        #Find a file name that does not already exist
        fileNumber = 1
        thisFileDir = './{}/TectonicEarth{}.npz'.format(self.npzSaveDirectory, fileNumber)
        while os.path.isdir(thisFileDir):
            fileNumber += 1
            thisFileDir = './{}/TectonicEarth{}.npz'.format(self.npzSaveDirectory, fileNumber)
        self.thisFileDir = thisFileDir
        
        #Save the file
        np.savez_compressed(thisFileDir, 
                timeHistory=self.timeHistory, 
                heightHistory=self.heightHistory, 
                tectonicDispHistory=self.tectonicDispHistory)
    
    #Load tectonic earth data from a saved NPZ file
    def loadDataFromNPZ(self, fileNumber):
        self.thisFileDir = './{}/TectonicEarth{}.npz'.format(self.npzSaveDirectory, fileNumber)
        npzFile = np.load(self.thisFileDir)
        self.timeHistory = npzFile['timeHistory']
        self.heightHistory = npzFile['heightHistory']
        self.tectonicDispHistory = npzFile['tectonicDispHistory']
        
        self.startTime = np.max(self.timeHistory)
        self.endTime = np.min(self.timeHistory)
        self.deltaTime = self.timeHistory[0] - self.timeHistory[1]
    
    #=================================================== Move Tectonic Plates =================================================
    #Create cylinder of earth such that vertices are all equally spaced apart
    #Overriding plates can then be identified as vertices being relatively closer to each other
    def createEarthCylinder(self):
        phiRes = self.phiResolution
        thetaRes = self.thetaResolution
        movedLonLat = self.movedLonLat
        northToSoutDist = np.max(movedLonLat[:, 1]) - np.min(movedLonLat[:, 1])
        cylinderRadius = thetaRes * northToSoutDist / (np.pi * phiRes * 2)
        cylinderXYZ = EarthAssist.cylindricalToCartesian(cylinderRadius, movedLonLat[:, 0], movedLonLat[:, 1])
        return cylinderXYZ, thetaRes   
        
    #Set the parameters "self.isCluster" and "self.clusterPointsNeighboursId"
    #These will represent regions of overriding/colliding plates, and will be used for interpolating scalars after a remesh
    def setRemeshClusters(self):
        cylinderXYZ, thetaRes = self.createEarthCylinder()
        
        #Run the clustering algorithm
        threshHoldDist = self.clusterThresholdProportion * 360 / thetaRes
        cluster = DBSCAN(eps=threshHoldDist, min_samples=self.minClusterSize).fit(cylinderXYZ)
        self.isCluster = (cluster.labels_ != -1)

        #Create KDTree to find nearest neighbours of each point in cluster
        pointsInClusterLonLat = cylinderXYZ[self.isCluster]
        clusterKDTree = cKDTree(pointsInClusterLonLat).query(pointsInClusterLonLat, k=self.numOfNeighbsForRemesh+1)
        self.clusterPointsNeighboursId = clusterKDTree[1]
    
    #Interpolate scalars after moving tectonic plates
    def interpolateScalars(self):
        self.setRemeshClusters()
        self.heightHistory.append(self.interpolateScalar(np.copy(self.heights)))
        self.oceanFloorAge = self.interpolateScalar(np.copy(self.oceanFloorAge))

    #Function based on interpolateHeights() from earth's remesh algorithm
    def interpolateScalar(self, scalar):
        scalarForRemesh = self.prepareScalarsForRemesh(scalar)
        movedLonLat = self.movedLonLat
        newScalar = griddata(movedLonLat, scalarForRemesh, self.lonLat)
        whereNAN = np.argwhere(np.isnan(newScalar))
        newScalar[whereNAN] = griddata(movedLonLat, scalarForRemesh, self.lonLat[whereNAN], method='nearest')
        return newScalar
        
    #To prepare scalars for interpolation, we set scalars of overlaping vertices to the maximum of their neighbours
    def prepareScalarsForRemesh(self, scalar):
        isCluster = self.isCluster
        clusterPointsNeighboursId = self.clusterPointsNeighboursId
        scalarsInCluster = scalar[isCluster]
        neighbourScalars = scalarsInCluster[clusterPointsNeighboursId[:, 1:]]
        scalar[isCluster] = np.max(neighbourScalars, axis=1)
        return scalar
    
    #Main function to call for moving plates and remeshing the sphere
    def movePlatesAndRemesh(self):
        movedEarthXYZ = self.movePlates(self.plateIds, self.rotations)
        movedLonLat = EarthAssist.cartesianToPolarCoords(movedEarthXYZ)
        self.movedLonLat = np.stack((movedLonLat[1], movedLonLat[2]), axis=1)
        self.interpolateScalars()
    
    #Move tectonic plates along the sphere by applying rotations to vertices with appropriate plate ids
    def movePlates(self, plateIds, rotations):
        newXYZ = np.copy(self.sphereXYZ)
        for idx in np.unique(plateIds):
            rot = rotations[idx]
            newXYZ[plateIds == idx] = rot.apply(newXYZ[plateIds == idx])
        return newXYZ
    
    #Get stage rotation data from pygplates and return a scipy rotation
    def getRotations(self, plateIds, time):
        rotations = {}
        for idx in np.unique(plateIds):
            stageRotation = self.rotationModel.get_rotation(int(time-self.deltaTime), int(idx), int(time))
            stageRotation = stageRotation.get_euler_pole_and_angle()
            axisLatLon = stageRotation[0].to_lat_lon()
            axis = EarthAssist.polarToCartesian(1, axisLatLon[1], axisLatLon[0])
            angle = stageRotation[1]
            rotations[idx] = R.from_quat(EarthAssist.quaternion(axis, angle))
        return rotations
    
    #Keep track of tectonic displacements for Gospl
    def createTectonicDisplacements(self):
        earthBeforeXYZ = self.getEarthXYZ(amplifier=1, iteration=-2)
        heightsAfter = self.heights
        radius = heightsAfter + self.earthRadius
        earthAfterXYZ = EarthAssist.polarToCartesian(radius, self.movedLonLat[:, 0], self.movedLonLat[:, 1])
        tectonicDisp = (earthAfterXYZ - earthBeforeXYZ) / (self.deltaTime * 1000000)
        self.tectonicDispHistory.append(tectonicDisp)
        
    #========================================== Reading From Data Sources =============================================
    #Read initial landscape data at specified time from file which is in the form of (lon, lat, height)
    @staticmethod
    def getInitialEarth(time, initialElevationFilesDir=None):
        if initialElevationFilesDir == None:
            initialElevationFilesDir = Earth.mainDirectory + '/PaleoDEMS'
        
        #Get path of initial landscape data file at specified time
        paleoDemsPath = Path(initialElevationFilesDir)
        
        #Try reading the specified elevation file, and raise an error if it does not exist
        try:
            initialLandscapePath = list(paleoDemsPath.glob('**/*%03.fMa.csv'%time))[0]
        except IndexError:
            raise IndexError("Unable to initialise earth. There is no initial topography data file at the specified time. Try changing earth's initial startTime variable.")
        
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

