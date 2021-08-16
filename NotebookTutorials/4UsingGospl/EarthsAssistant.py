
import numpy as np
import pyvista as pv

class EarthAssist:
    
    #Coordinate transformation from spherical polar to cartesian
    @staticmethod
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
    @staticmethod
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
    @staticmethod
    def cartesianToCylindrical(X, Y, Z):
        r = (X**2 + Y**2)**0.5
        theta = np.arctan2(Y, X)
        return np.stack((r, theta, Z), axis=1)

    #Coordinate transformation functions from cylindrical polar coordinates to cartesian
    @staticmethod
    def cylindricalToCartesian(r, theta, Z, useDegrees=True):
        if useDegrees == True:
            theta = np.radians(theta+180.)
        X = r * np.cos(theta)
        Y = r * np.sin(theta)
        return np.stack((X, Y, Z), axis=1)

    #Returns a rotation quaternion
    @staticmethod
    def quaternion(axis, angle):
        return [np.sin(angle/2) * axis[0], 
                np.sin(angle/2) * axis[1], 
                np.sin(angle/2) * axis[2], 
                np.cos(angle/2)]
    
    #Normalizes the height map or optionally brings the heightmap within specified height limits
    @staticmethod
    def normalizeArray(A, minValue=0.0, maxValue=1.0):
        A = A - min(A)
        A = A / (max(A) - min(A))
        return A * (maxValue - minValue) + minValue
    
    #Gaussian function used for profile of diverge lowering
    @staticmethod
    def gaussian(x, mean=0, variance=0.25):
        return np.exp(-((x - mean)**2) / variance)

    #Sigmoid function used for  height transfer of diverge lowering
    @staticmethod
    def sigmoid(x, centre=-0.1, steepness=6):
        return 1 / (1 + np.exp(-(x - centre) * steepness))
    
    #Creates a mesh object of the tectonic displacements vector field for visualizations
    #We begin by normalizing the vectors so that their arrow sizes are consistent
    @staticmethod
    def createTectonicVectorMesh(vertices, displacementXYZ, ignoreEveryNverts=50, vectorScale=600000):
        dispMagnitude = np.linalg.norm(displacementXYZ, axis=1)
        dispDirection = displacementXYZ / dispMagnitude[:, np.newaxis]
        dispMagnitude /= np.max(dispMagnitude)
        
        #Create an arrow template and place it at various locations just above earth's mesh
        arrow = pv.Arrow()
        reduceIndex = (np.arange(vertices.shape[0]) % ignoreEveryNverts == 0)
        vectorLocations = pv.PolyData(vertices[reduceIndex] * 1.02)
        vectorLocations['dispMag'] = dispMagnitude[reduceIndex]
        vectorLocations['dispDir'] = dispDirection[reduceIndex]
        forceVectors = vectorLocations.glyph(orient="dispDir", scale="dispMag", factor=vectorScale, geom=arrow)
        return forceVectors

