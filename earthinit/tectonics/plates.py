import pygplates
import numpy as np
import pandas as pd
from scipy import ndimage
from time import process_time
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as scpRot

from .boundary import PlateBoundary as _PlateBoundary
from .boundary import SubductionBoundary as _SubductionBoundary
from .boundary import RidgeBoundary as _RidgeBoundary


class PlateInfo(object):
    """
    This class is the main entry point to interact with plate evolution. It is used to compute the uplift, subsidence in converging and diverging regions as well as the informations regarding plate topologies.

    """

    def __init__(self):

        # Get plates information at start time
        t0 = process_time()
        if self.paleoVelocityPath is not None:
            self.updatePlates()
        else:
            if self.interpZ is not None:
                self.elev = self.interpZ.copy()
        if self.verbose:
            print(
                "\n-- Create plate informations (%0.02f seconds)"
                % (process_time() - t0),
                flush=True,
            )

    def updatePlates(self):
        """
        The following function encompasses all the required functions used for moving plates and assigning plate informations as the model evolve through time.
        """

        # Define plate IDs and rotations using pygplates information
        t0 = process_time()
        self._getPlateIDs()
        self._getRotations()
        if self.verbose:
            print(
                "Get plate indices & rotations (%0.02f seconds)"
                % (process_time() - t0),
                flush=True,
            )

        if self.paleoDemForce or self.tecForce is not None or not self.tectoEarth:
            return

        if self.tNow > self.tEnd:
            # Get plate centres for transfer function calculation
            t0 = process_time()
            self._getPlateCentres()
            if self.verbose:
                print(
                    "Get plate centres (%0.02f seconds)" % (process_time() - t0),
                    flush=True,
                )

            # Compute plate boundaries to estimate which of the plates are converging or diverging
            t0 = process_time()
            self._plateBoundaries()
            if self.verbose:
                print(
                    "Get plate boundaries (%0.02f seconds)" % (process_time() - t0),
                    flush=True,
                )

            # Get the information requires for converging plates
            t0 = process_time()
            self._getSubductionBoundsForEachPlateId()
            self._setCollisionSpeeds()
            if self.verbose:
                print(
                    "Get collision speed for each plate (%0.02f seconds)"
                    % (process_time() - t0),
                    flush=True,
                )

            # From transfer function estimate the related uplift
            t0 = process_time()
            self.getUplifts()
            if self.verbose:
                print(
                    "Get uplift for collision regions (%0.02f seconds)"
                    % (process_time() - t0),
                    flush=True,
                )

            # For diverging plates estimate the related subsidence
            t0 = process_time()
            self.getDivergeLowering()
            if self.verbose:
                print(
                    "Get subsidence for diverging regions (%0.02f seconds)"
                    % (process_time() - t0),
                    flush=True,
                )

        return

    def _getRotations(self):
        """
        To move tectonic plates, we create a rotation quaternion for each plate, and apply rotations to all vertices based on their plate ids.

        We use the library pygplates to assign plate ids to vertices. Pygplates requires a list of point features for each vertex on our sphere and assigns plate ids to those.
        """

        rotationModel = pygplates.RotationModel(self.rotationsDirectory)

        self.rotations = {}
        for plateId in np.unique(self.plateIds):
            if self.reverse:
                stageRotation = rotationModel.get_rotation(
                    int(self.tNow + self.dt), int(plateId), int(self.tNow)
                )

            else:
                stageRotation = rotationModel.get_rotation(
                    int(self.tNow - self.dt), int(plateId), int(self.tNow)
                )
            stageRotation = stageRotation.get_euler_pole_and_angle()

            axisLatLon = stageRotation[0].to_lat_lon()
            axis = self._polarToCartesian(1, axisLatLon[1], axisLatLon[0])
            angle = stageRotation[1]
            self.rotations[plateId] = scpRot.from_quat(self._quaternion(axis, angle))

        return

    def _getPlateIDs(self):
        """
        To specify which tectonic plate a particular vertex on our sphere belongs to, we create a list of Plate Ids where each vertex is given a number based on which plate they belong to.
        """

        if self.interpZ is not None:
            self.elev = self.interpZ.copy()

        # Read plate IDs from gPlates exports
        velfile = self.paleoVelocityPath + "/vel" + str(int(self.tNow)) + "Ma.xy"
        data = pd.read_csv(
            velfile,
            sep=r"\s+",
            engine="c",
            header=None,
            na_filter=False,
            dtype=float,
            low_memory=False,
        )
        data = data.drop_duplicates().reset_index(drop=True)
        llvel = data.iloc[:, 0:2].to_numpy()
        gplateID = data.iloc[:, -1].to_numpy().astype(int)
        vtree = cKDTree(llvel)
        dist, ids = vtree.query(self.lonlat, k=1)
        self.plateIds = gplateID[ids]

        return

    def _plateBoundaries(self):
        """
        This function defines the list of plate boundaries at each time step.
        """

        # Use pygplates to get shared boundary sections at specified time
        resolvedTopologies, sharedBoundarySections = [], []
        rotationModel = pygplates.RotationModel(self.rotationsDirectory)
        pygplates.resolve_topologies(
            self.platePolygonsDirectory,
            rotationModel,
            resolvedTopologies,
            self.tNow,
            sharedBoundarySections,
        )

        # Loop through all shared plate boundaries
        plateBoundaries = []
        for sharedBound in sharedBoundarySections:

            # Identify which type of boundary this is
            boundType = sharedBound.get_feature().get_feature_type()
            isSubduction = boundType == pygplates.FeatureType.gpml_subduction_zone
            isOceanicRidge = boundType == pygplates.FeatureType.gpml_mid_ocean_ridge
            if _PlateBoundary.ignoreThisBoundary(sharedBound):
                continue

            # Create plate boundary object of appropriate type and append to list of plate boundaries
            if isSubduction:
                plateBoundaries.append(_SubductionBoundary(self.radius, sharedBound))
            elif isOceanicRidge:
                plateBoundaries.append(_RidgeBoundary(self.radius, sharedBound))
            else:
                plateBoundaries.append(_PlateBoundary(self.radius, sharedBound))
        self.pBounds = plateBoundaries.copy()

        lineCentres, linePoints = _PlateBoundary.prepareLines(self.pBounds)
        distIds = cKDTree(lineCentres).query(self.xyz)[1]
        closestLinePoints = linePoints[distIds]
        self.distToBound = _PlateBoundary.getDistsToLinesSeg(
            self.xyz, closestLinePoints
        )

        return

    def _getPlateCentres(self):
        """
        This function creates a dictionary containing plates centres coordinates.
        """

        self.plateCentres = {}
        uniqueIds = np.unique(self.plateIds)
        for i in range(uniqueIds.shape[0]):
            plateXYZ = self.xyz[self.plateIds == uniqueIds[i]]
            self.plateCentres[uniqueIds[i]] = (
                np.sum(plateXYZ, axis=0) / plateXYZ.shape[0]
            )

        return

    @staticmethod
    def getCollisionsSpeeds(
        boundXYZ, sharedPlateIds, rotations, plateCentres, deltaTime
    ):
        """
        This function calculates the speed of collision between converging plates. The function identifies if a specif boundary segment belongs to a converging or diverging boundary and set the speed to positive or negative value accordingly.

        .. note::

            Depending on the mesh resolution, some tectonic plates might be too small to have any vertices representing them. Consequently, some `sharedIds` might have undefined rotations or plate centres. In such case, the speed of the plate is set to 0 and the specific plates IDs are ignored.

        """

        speeds = np.zeros(boundXYZ.shape[0])
        directions = np.zeros((boundXYZ.shape[0], 3))

        for i in range(boundXYZ.shape[0]):
            bXYZ = boundXYZ[i]
            shareId = sharedPlateIds[i]

            # Ignore small plates
            if not np.all(np.isin(shareId, list(rotations.keys()))):
                continue

            # Calculate speed based on distance moved by rotations
            rot0 = rotations[shareId[0]]
            rot1 = rotations[shareId[1]]
            movedXyz0 = rot0.apply(bXYZ)
            movedXyz1 = rot1.apply(bXYZ)
            velocity = (movedXyz1 - movedXyz0) / deltaTime
            speed = np.linalg.norm(velocity)
            if speed != 0:
                directions[i] = velocity / speed

            # Assign speed to positive or negative value based on converging/diverging boundaries
            cent0 = plateCentres[shareId[0]]
            cent1 = plateCentres[shareId[1]]
            centDist = np.linalg.norm(cent1 - cent0)
            centDistAfter = np.linalg.norm(rot1.apply(cent1) - rot0.apply(cent0))
            if centDist < centDistAfter:
                speed = -speed
            speeds[i] = speed

        return speeds, directions

    def _getSubductionBoundsForEachPlateId(self):
        """
        This function creates a dictionary with plate ids as keys and the plate's subduction boundary as values.

        """

        self.idToSubBound = {}
        for bound in self.pBounds:

            # List to store plate ids of overriding plates for this boundary
            overId = []

            # Append plate ids of overiding plates
            if bound.boundType == 1:
                for idx in bound.overPlateId:
                    overId.append(idx)

            # Append plate ids of both sides of an orogenic belt
            elif bound.gpmlBoundType == "gpml:OrogenicBelt":
                for sIds in bound.sharedPlateIds:
                    for idx in sIds:
                        overId.append(idx)

            # We append this plate boundary to dictionary values with appropriate keys (plateIds)
            overId = np.unique(overId)
            for idx in overId:
                if idx not in list(self.idToSubBound.keys()):
                    self.idToSubBound[idx] = []
                self.idToSubBound[idx].append(bound)

        return

    def _setCollisionSpeeds(self):
        """
        This function sets the collision speed attribute for points in the plate boundary objects.

        """

        for bound in self.pBounds:
            for i in range(bound.XYZ.shape[0] - 1):

                point1 = bound.XYZ[i]
                point2 = bound.XYZ[i + 1]
                bound.lineCentres[i] = (point1 + point2) / 2
                bound.linePoints[i, 0] = point1
                bound.linePoints[i, 1] = point2
            speeds, directions = self.getCollisionsSpeeds(
                bound.lineCentres,
                bound.sharedPlateIds[:-1],
                self.rotations,
                self.plateCentres,
                self.dt,
            )
            bound.collisionSpeed = speeds
            bound.speedDirection = directions

        return
