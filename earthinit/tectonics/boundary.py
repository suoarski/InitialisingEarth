import numpy as np
from numba import jit


def polarToCartesian(radius, theta, phi, useLonLat=True):
    """
    Coordinate transformation from polar to cartesian.

    :arg radius: radius
    :arg theta: polar angle
    :arg phi: azimuthal angle
    :arg useLonLat: boolean set to True when lon/lat coordinates are used.

    :return: X,Y,Z spherical coordinates

    """

    if useLonLat is True:
        theta, phi = np.radians(theta + 180.0), np.radians(90.0 - phi)
    X = radius * np.cos(theta) * np.sin(phi)
    Y = radius * np.sin(theta) * np.sin(phi)
    Z = radius * np.cos(phi)

    if type(X) == np.ndarray:
        return np.stack((X, Y, Z), axis=1)
    else:
        return np.array([X, Y, Z])


# Create a general plate boundary class
class PlateBoundary:
    """
    This class creates a general plate boundary class that is used to store information about different tectonic forcing conditions.

    The `PlateBoundary` class has two child-classes:

    - `SubductionBoundary` and
    - `RidgeBoundary`

    The class is initialised with a shareBound object obtained from the `pygplates` library.

    .. important::

        The attribute `sharedPlateIds` contains the plate ids of neighbouring plates for each point on the plate boundary. In most cases, `pygplates` provides two plates as the neighbouring plates, however occasionally it will provide a different number of neighbouring plates. If pygplates doesn't provide exactly two neighbouring plates, then we will not include that particular plate boundary section.

    """

    def __init__(self, radius, sharedBound):

        # Extract relevant data from sharedBounds
        lon, lat, sharedPlateIds = [], [], []
        for sharedSubSection in sharedBound.get_shared_sub_segments():
            latLon = sharedSubSection.get_resolved_geometry().to_lat_lon_array()

            # Get ids of neighbouring plates
            sharedId = []
            for i in sharedSubSection.get_sharing_resolved_topologies():
                idx = i.get_resolved_feature().get_reconstruction_plate_id()
                sharedId.append(idx)

            # Ignore plate boundaries with not exactly two neighbouring plates
            twoSharedIds = len(sharedId) == 2
            if not twoSharedIds:
                continue

            # Append relevant data to lists
            for i in range(len(latLon)):
                lon.append(latLon[i, 1])
                lat.append(latLon[i, 0])
                sharedPlateIds.append(sharedId)

        # Store data as class attributes
        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.XYZ = polarToCartesian(radius, self.lon, self.lat)
        self.sharedPlateIds = np.array(sharedPlateIds)
        self.boundType = 0
        self.gpmlBoundType = str(sharedBound.get_feature().get_feature_type())

        # These attributes will be set later
        self.lineCentres = np.zeros((self.XYZ.shape[0] - 1, 3))
        self.linePoints = np.zeros((self.XYZ.shape[0] - 1, 2, 3))
        self.collisionSpeed = np.zeros(self.XYZ.shape[0] - 1)
        self.speedDirection = np.zeros((self.XYZ.shape[0] - 1, 3))

    @staticmethod
    def ignoreThisBoundary(sharedBound):
        """
        Check if a specific plate boundary has more then two neighbouring plates. If this is the case flag it.
        """

        ignoreThis = True
        for s in sharedBound.get_shared_sub_segments():
            if len(s.get_sharing_resolved_topologies()) == 2:
                ignoreThis = False

        return ignoreThis

    def prepareLines(pBounds):
        """
        This function is used when calculating the distance of each point in a particular plate to its boundaries.

        It returns the center between 2 points forming a segment of a specific plate boundary and the points coordinates.
        """

        lineCentres, linePoints = [], []
        for bound in pBounds:
            for i in range(bound.XYZ.shape[0] - 1):
                point1 = bound.XYZ[i]
                point2 = bound.XYZ[i + 1]
                lineCentres.append((point1 + point2) / 2)
                linePoints.append([point1, point2])
        return np.array(lineCentres), np.array(linePoints)

    @staticmethod
    @jit(nopython=True)
    def getDistsToLinesSeg(sphereXYZ, closestLinePoints):
        """
        This function computes the distance from line segment based on the dot product.

        .. note::

            Since this function is slow, we use the `@jit(nopython=True)` decorator from `numba` library to significantly speed things up. Although this decorator poses many restrictions on the content of the function such that we can not use many libraries apart from numpy, it speeds up our calculation by 10 times the speed.

        """

        distToBound = np.zeros(sphereXYZ.shape[0])
        for i in range(sphereXYZ.shape[0]):
            linePoints = closestLinePoints[i]
            xyz = sphereXYZ[i]

            # Append distance from vertex 0
            v = linePoints[1] - linePoints[0]
            w = xyz - linePoints[0]
            if np.dot(w, v) <= 0:
                distToZero = np.linalg.norm(linePoints[0] - xyz)
                distToBound[i] = distToZero

            # Append distance from vertex 1
            elif np.dot(v, v) <= np.dot(w, v):
                distToOne = np.linalg.norm(linePoints[1] - xyz)
                distToBound[i] = distToOne

            # Append distance from somewhere in the line centre
            else:
                numerator = np.linalg.norm(
                    np.cross(linePoints[1] - xyz, linePoints[1] - linePoints[0])
                )
                denominator = np.linalg.norm(linePoints[1] - linePoints[0])
                distToLine = numerator / denominator
                distToBound[i] = distToLine

        return distToBound


class SubductionBoundary(PlateBoundary):
    """
    This child class inherits from the `PlateBoundary` class and is used to specify subduction plate boundaries.
    """

    def __init__(self, radius, sharedBound):

        # Run the parent's initialization
        PlateBoundary.__init__(self, radius, sharedBound)

        # Get plate ids of overriding plate and subducting plates
        overPlateId, subPlateId = [], []
        for sharedSubSection in sharedBound.get_shared_sub_segments():
            overAndSubPlates = sharedSubSection.get_overriding_and_subducting_plates(
                True
            )
            if overAndSubPlates is not None:
                overridingPlate, subductingPlate, subduction_polarity = overAndSubPlates
                overPlateId.append(
                    overridingPlate.get_feature().get_reconstruction_plate_id()
                )
                subPlateId.append(
                    subductingPlate.get_feature().get_reconstruction_plate_id()
                )

        # Save data
        self.overPlateId = np.unique(overPlateId)
        self.subPlateId = np.unique(subPlateId)
        self.boundType = 1


class RidgeBoundary(PlateBoundary):
    """
    This child class inherits from the `PlateBoundary` class and is used to specify ridge plate boundaries.
    """

    def __init__(self, radius, sharedBound):
        PlateBoundary.__init__(self, radius, sharedBound)
        self.boundType = 2
