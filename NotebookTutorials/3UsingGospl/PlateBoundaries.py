
import pygplates
import numpy as np
from numba import jit
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from EarthsAssistant import EarthAssist
from scipy.spatial.distance import cdist

#================================================== Plate Boundary Classes ========================================================
#Create a general plate boundary class
class PlateBoundary:
    def __init__(self, sharedBound, earthRadius=6378.137):
        
        #Extract relevant data from sharedBounds
        lon, lat, sharedPlateIds = [], [], []
        for sharedSubSection in sharedBound.get_shared_sub_segments():
            latLon = sharedSubSection.get_resolved_geometry().to_lat_lon_array()
            
            #Get ids of neighbouring plates
            sharedId = []
            for i in sharedSubSection.get_sharing_resolved_topologies():
                idx = i.get_resolved_feature().get_reconstruction_plate_id()
                sharedId.append(idx)
            
            #Ignore plate boundaries with not exactly two neighbouring plates
            twoSharedIds =  (len(sharedId) == 2)
            if not twoSharedIds:
                continue
            
            #Append relevant data to lists
            for i in range(len(latLon)):
                lon.append(latLon[i, 1])
                lat.append(latLon[i, 0])
                sharedPlateIds.append(sharedId)
        
        #Store data as class attributes
        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.XYZ = EarthAssist.polarToCartesian(earthRadius, self.lon, self.lat)
        self.sharedPlateIds = np.array(sharedPlateIds)
        self.boundType = 0
        self.gpmlBoundType = str(sharedBound.get_feature().get_feature_type())
        
        #These attributes will be set later
        self.lineCentres = np.zeros((self.XYZ.shape[0] - 1, 3))
        self.linePoints = np.zeros((self.XYZ.shape[0] - 1, 2, 3))
        self.collisionSpeed = np.zeros(self.XYZ.shape[0] - 1)
        self.speedDirection = np.zeros((self.XYZ.shape[0] - 1, 3))

#Create child class for subduction plate boundaries
class SubductionBoundary(PlateBoundary):
    def __init__(self, sharedBound, earthRadius=6378.137):
        
        #Run the parent's initialization
        PlateBoundary.__init__(self, sharedBound, earthRadius=earthRadius)
        
        #Get plate ids of overriding plate and subducting plates
        overPlateId, subPlateId = [], []
        for sharedSubSection in sharedBound.get_shared_sub_segments():
            overAndSubPlates = sharedSubSection.get_overriding_and_subducting_plates(True)
            if (overAndSubPlates != None):
                overridingPlate, subductingPlate, subduction_polarity = overAndSubPlates
                overPlateId.append(overridingPlate.get_feature().get_reconstruction_plate_id())
                subPlateId.append(subductingPlate.get_feature().get_reconstruction_plate_id())
        
        #Save data
        self.overPateId = np.unique(overPlateId)
        self.subPlateId = np.unique(subPlateId)
        self.boundType = 1

#Create child class for ridge plate boundaries
class RidgeBoundary(PlateBoundary):
    def __init__(self, sharedBound, earthRadius=6378.137):
        PlateBoundary.__init__(self, sharedBound, earthRadius=earthRadius)
        self.boundType = 2

#================================================== Boundaries Main Class ========================================================
#================================================== Initialization ===============================================================
#This class will contain a list of plate boundary objects at a specified time. It also contains most of the algorithms discussed in the second notebook
class Boundaries:
    def __init__(self, 
                time,
                earth,
                plateIds,
                rotations,
                baseUplift = 2,
                distTransRange = 1000, 
                numToAverageOver = 10
        ):
        
        #Set class attributes
        self.time = int(time)
        self.earth = earth
        self.plateIds = plateIds
        self.rotations = rotations
        self.baseUplift = baseUplift
        self.distTransRange = distTransRange
        self.numToAverageOver = numToAverageOver
        
        self.earthRadius = earth.earthRadius
        self.rotationsDirectory = earth.rotationsDirectory
        self.platePolygonsDirectory = earth.platePolygonsDirectory
        
        self.plateCentres = self.getPlateCentres()
        self.plateBoundaries = self.createPlateBoundariesAtTime()
        self.idToSubBound = self.getSubductionBoundsForEachPlateId()
        self.distanceTransferFunction = self.getDistanceTransferFunction()
        self.setCollisionSpeeds()
    
    #Create function that returns a list of plate boundaries at specified times
    def createPlateBoundariesAtTime(self):
        
        #Use pygplates to get shared boundary sections at specified time
        resolvedTopologies, sharedBoundarySections = [], []
        rotationModel = pygplates.RotationModel(self.rotationsDirectory)
        pygplates.resolve_topologies(self.platePolygonsDirectory, rotationModel, resolvedTopologies, self.time, sharedBoundarySections)
        
        #Loop through all shared plate boundaries
        plateBoundaries = []
        for sharedBound in sharedBoundarySections:

            #Identify which type of boundary this is
            boundType = sharedBound.get_feature().get_feature_type()
            isSubduction = boundType == pygplates.FeatureType.gpml_subduction_zone
            isOceanicRidge = boundType == pygplates.FeatureType.gpml_mid_ocean_ridge
            if self.ignoreThisBoundary(sharedBound):
                continue

            #Create plate boundary object of appropriate type and append to list of plate boundaries
            if isSubduction:
                plateBoundaries.append(SubductionBoundary(sharedBound, earthRadius=self.earthRadius))
            elif isOceanicRidge:
                plateBoundaries.append(RidgeBoundary(sharedBound, earthRadius=self.earthRadius))
            else:
                plateBoundaries.append(PlateBoundary(sharedBound, earthRadius=self.earthRadius))
        return plateBoundaries


    #Since we are ignoring plate boundaries with not exactly two neighbouring plates,
    #Some plate boundaries will have no coordinates, so we ignore those
    @staticmethod
    def ignoreThisBoundary(sharedBound):
        ignoreThis = True
        for s in sharedBound.get_shared_sub_segments():
            if len(s.get_sharing_resolved_topologies()) == 2:
                ignoreThis = False
        return ignoreThis
    
    #================================================== Distance And Speeds ===============================================================
    #We set the collision speed attribute for points in our plate boundary objects
    def setCollisionSpeeds(self):
        for bound in self.plateBoundaries:
            for i in range(bound.XYZ.shape[0]-1):
                point1 = bound.XYZ[i]
                point2 = bound.XYZ[i+1]
                bound.lineCentres[i] = (point1 + point2) / 2
                bound.linePoints[i, 0] = point1
                bound.linePoints[i, 1] = point2
            speeds, directions = self.getCollisionsSpeeds(
                bound.lineCentres, 
                bound.sharedPlateIds[:-1], 
                self.rotations, 
                self.plateCentres, 
                self.earth.deltaTime)
            bound.collisionSpeed = speeds
            bound.speedDirection = directions
        #return plateBounds
    
    #Calculate the speed of collisions
    @staticmethod
    def getCollisionsSpeeds(boundXYZ, sharedPlateIds, rotations, plateCentres, deltaTime):
        speeds = np.zeros(boundXYZ.shape[0])
        directions = np.zeros((boundXYZ.shape[0], 3))
        for i in range(boundXYZ.shape[0]):
            bXYZ = boundXYZ[i]
            shareId = sharedPlateIds[i]
            
            #Some tectonic plates are so small that we don't have any vertices representing it
            #Therefore, some sharedIds might not have rotations or plate centres defined for it
            #In this case we simply set the speed to 0 and ignore the rest of this function
            if not np.all(np.isin(shareId, list(rotations.keys()))):
                continue
            
            #Calculate speed based on distance moved by rotations
            rot0 = rotations[shareId[0]]
            rot1 = rotations[shareId[1]]
            movedXyz0 = rot0.apply(bXYZ)
            movedXyz1 = rot1.apply(bXYZ)
            velocity = (movedXyz1 - movedXyz0) / deltaTime
            speed = np.linalg.norm(velocity)
            if speed != 0:
                directions[i] = velocity / speed
            
            #Identify if this boundary segment belongs to a converging or diverging boundary
            #Then set speed to positive or negative value accordingly
            cent0 = plateCentres[shareId[0]]
            cent1 = plateCentres[shareId[1]]
            centDist = np.linalg.norm(cent1 - cent0)
            centDistAfter = np.linalg.norm(rot1.apply(cent1) - rot0.apply(cent0))
            if centDist < centDistAfter:
                speed = -speed
            speeds[i] = speed
        return speeds, directions
    
    #Create dictionary with plate ids as keys and the plate's subduction boundary as values
    def getSubductionBoundsForEachPlateId(self):
        idToSubBound = {}
        for bound in self.plateBoundaries:
            
            #List to store plate ids of overriding plates for this boundary
            overId = []
            
            #Append plate ids of overiding plates
            if bound.boundType == 1:
                for idx in bound.overPateId:
                    overId.append(idx)
            
            #Append plate ids of both sides of an orogenic belt
            elif bound.gpmlBoundType == 'gpml:OrogenicBelt':
                for sIds in bound.sharedPlateIds:
                    for idx in sIds:
                        overId.append(idx)
            
            #We append this plate boundary to dictionary values with appropriate keys (plateIds)
            overId = np.unique(overId)
            for idx in overId:
                if (idx not in list(idToSubBound.keys())):
                    idToSubBound[idx] = []
                idToSubBound[idx].append(bound)
        return idToSubBound
    
    
    
    
    
    
    #Get subduction uplifts using speed and distance transfers.
    def getUplifts(self):
        speedTransfer, distTransfer = self.getTransfers()
        uplifts = distTransfer * speedTransfer
        uplifts[uplifts <= 0.0001] = 0.0001
        uplifts /= np.max(uplifts)
        uplifts *= self.baseUplift
        return uplifts
    
    #Calculates transfers for all plates on the sphere
    def getTransfers(self):
        plateIds = self.plateIds
        sphereXYZ = self.earth.sphereXYZ
        speedTransfer = np.zeros(sphereXYZ.shape[0])
        angleTransfer = np.zeros(sphereXYZ.shape[0])
        distToSubBounds = np.zeros(sphereXYZ.shape[0])
        
        for idx in np.unique(plateIds):
            if (idx not in self.idToSubBound.keys()):
                continue
            
            subBounds = self.idToSubBound[idx]
            xyzOnThisPlate = sphereXYZ[plateIds == idx]
            speedTrans, angleTrans, distToBound = self.getTransfersForThisPlate(subBounds, xyzOnThisPlate)
            speedTransfer[plateIds==idx] = np.sum(speedTrans, axis=0) / self.numToAverageOver
            angleTransfer[plateIds==idx] = np.sum(angleTrans, axis=0) / self.numToAverageOver
            distToSubBounds[plateIds==idx] = distToBound
        
        #Calculate the distance transfer
        dTransInput = distToSubBounds / (self.distTransRange * angleTransfer + 0.01)
        dTransInput[dTransInput>=100] = 100
        dTransInput[dTransInput<= -100] = -100
        
        distTransfer = self.distanceTransferFunction(dTransInput) * (dTransInput <= 1.01)
        return speedTransfer, distTransfer
    
    #Calculates transfers one plate at a time
    def getTransfersForThisPlate(self, subBounds, xyzOnThisPlate):
        boundXYZ, boundSpeed, bCollisionsDirection, linePoints, lineLengths = self.getSpeedsData(subBounds)
        numToAverageOver = self.numToAverageOver
        
        #Get distances and the index array distIds
        distIds = cKDTree(boundXYZ).query(xyzOnThisPlate, k=numToAverageOver)[1]
        distIds[distIds >= boundXYZ.shape[0]] = boundXYZ.shape[0]-1 #Avoid 'index out of bounds error'
        closestLinePoints = linePoints[distIds]
        distToBound = self.getDistsToLinesSeg(xyzOnThisPlate, closestLinePoints[:, 0])
        directionToBound = self.getDirectionToBound(xyzOnThisPlate, closestLinePoints)
        
        #'distIds' contains the indices of the N closest boundaries for each vertex XYZ on this plate
        boundSpds = boundSpeed[distIds]
        lineLngths = lineLengths[distIds]
        boundDirs = bCollisionsDirection[distIds]
        
        #Main transfer calculations
        speedTrans = np.zeros((numToAverageOver, xyzOnThisPlate.shape[0]))
        angleTrans = np.zeros((numToAverageOver, xyzOnThisPlate.shape[0]))
        for j in range(numToAverageOver):
            cosAngle = np.einsum('ij,ij->i', directionToBound[:, j], boundDirs[:, j]) #Dot product
            speedTrans[j] = np.abs(boundSpds[:, j] * cosAngle)
            angleTrans[j] = np.abs(cosAngle)
        return speedTrans, angleTrans, distToBound
    
    #Given a list of subduction boundaries, we extract relevant data and return as arrays
    @staticmethod
    def getSpeedsData(subBounds):
        boundXYZ, boundSpeed, boundDirection, linePoints = [], [], [], []
        for bound in subBounds:
            for i in range(bound.lineCentres.shape[0]):
                boundXYZ.append(bound.lineCentres[i])
                boundSpeed.append(bound.collisionSpeed[i])
                boundDirection.append(bound.speedDirection[i])
                linePoints.append(bound.linePoints[i])
        boundSpeed = np.array(boundSpeed)
        boundSpeed[boundSpeed < 0] = 0
        linePoints = np.array(linePoints)
        lineLengths = np.linalg.norm(linePoints[:, 0, :] - linePoints[:, 1, :], axis=1)
        return np.array(boundXYZ), boundSpeed, np.array(boundDirection), linePoints, lineLengths
    
    def getDistanceTransferFunction(self):
        #Template curve for the distance transfer for subduction uplift
        distTransPoints = np.array([
            [-100, 0.0],
            [-50, 0],
            [-10, 0],
            [-1.0, 0.0],
            [-0.101, 0.0],
            [-0.1, 0.0],
            [0, 0.4],
            [0.19, 1.0],
            [0.21, 1.0],
            [0.5, 0.5],
            [0.99, 0.0],
            [1.0, 0.0],
            [5.0, 0.0],
            [50.0, 0.0],
            [100.0, 0.0]
            ])
        return interp1d(distTransPoints[:, 0], distTransPoints[:, 1], kind='quadratic')
    
    #Get distance from line segments
    @staticmethod
    @jit(nopython=True)
    def getDistsToLinesSeg(sphereXYZ, closestLinePoints):
        distToBound = np.zeros(sphereXYZ.shape[0])
        for i in range(sphereXYZ.shape[0]):
            linePoints = closestLinePoints[i]
            xyz = sphereXYZ[i]

            #Append distance from vertex 0
            v = linePoints[1] - linePoints[0]
            w = xyz - linePoints[0]
            if np.dot(w, v) <= 0:
                distToZero = np.linalg.norm(linePoints[0] - xyz)
                distToBound[i] = distToZero

            #Append distance from vertex 1  
            elif np.dot(v, v) <= np.dot(w, v):
                distToOne = np.linalg.norm(linePoints[1] - xyz)
                distToBound[i] = distToOne

            #Append distance from somewhere in the line centre
            else:
                numerator = np.linalg.norm(np.cross(linePoints[1] - xyz, linePoints[1] - linePoints[0]))
                denominator = np.linalg.norm(linePoints[1] - linePoints[0])
                distToLine = numerator / denominator
                distToBound[i] = distToLine
        return distToBound
    
    #Get distance from line segments
    @staticmethod
    @jit(nopython=True)
    def getDirectionToBound(XYZ, closestLinePoints):
        directionToBound = np.zeros((XYZ.shape[0], closestLinePoints.shape[1], 3))
        for i in range(XYZ.shape[0]):
            linePoints = closestLinePoints[i]
            xyz = XYZ[i]

            #Get v and w to identify which case we are in
            v = linePoints[:, 1] - linePoints[:, 0]
            w = xyz - linePoints[:, 0]
            
            #Loop through each point to average over (later on)
            for j in range(w.shape[0]):
                
                #Case 1
                if np.dot(w[j], v[j]) <= 0:
                    direction = linePoints[j, 0] - xyz
                    directionToBound[i, j] = direction / np.linalg.norm(direction)

                #Case 2  
                elif np.dot(v[j], v[j]) <= np.dot(w[j], v[j]):
                    direction = linePoints[j, 1] - xyz
                    directionToBound[i, j] = direction / np.linalg.norm(direction)

                #Case 3
                else:
                    #direction = np.cross(xyz, linePoints[j, 1] - linePoints[j, 0])
                    direction = np.cross(linePoints[j, 0], linePoints[j, 1] - linePoints[j, 0])
                    directionToBound[i, j] = direction / np.linalg.norm(direction)
        return directionToBound
    
    #Create a dictionary containing plate Ids as keys and plate centres as values
    def getPlateCentres(self):
        plateCentres = {}
        uniqueIds = np.unique(self.plateIds)
        for i in range(uniqueIds.shape[0]):
            plateXYZ = self.earth.sphereXYZ[self.plateIds==uniqueIds[i]]
            plateCentres[uniqueIds[i]] = np.sum(plateXYZ, axis=0) / plateXYZ.shape[0]
        return plateCentres
    
    #================================================== Functions for Visualizations ========================================================
    def getBoundaryLines(self):
    
        #Variables used to create the line pyvista object
        XYZ, lineConnectivity, bType = [], [], []
        
        #Counter for keeping track of how many vertices we have used
        xyzCount = 0
        
        #Loop through all plate boundaries
        for bound in self.plateBoundaries:
            
            #Create lineID for defining line connectivity
            numOfPoints = len(bound.XYZ)
            lineConnectivity.append(numOfPoints)
            lineID = np.arange(numOfPoints) + xyzCount
            
            #Loop through points in plate boundary and append to arrays
            for i in range(numOfPoints):
                lineConnectivity.append(lineID[i])
                XYZ.append(bound.XYZ[i])
                bType.append(bound.boundType)
            xyzCount += numOfPoints
            
        #Create the line mesh
        lineMesh = pv.PolyData(np.array(XYZ), lines=lineConnectivity)
        lineMesh['boundType'] = np.array(bType)
        return lineMesh