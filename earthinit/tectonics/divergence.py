import numpy as np
from scipy.spatial import cKDTree

from .boundary import PlateBoundary as _PlateBoundary


class Divergence(object):
    r"""
    This class encapsulates the functions used to approximate subsidense in diverging regions.

    The amount of subsidense is defined as `divergeLowering`, which will depend on the following transfers:

    1. **Distance Transfer**: the distance a vertex is to the closest diverging plate boundary. Vertices that are close to a plate boundary should be lowered more than vertices far away from the boundaries, and so we will pass the distances through a gaussian shaped function to calculate the distance contribution to diverge lowering. Given the mean $\mu$ and variance $\sigma$, our gaussian is defined by:
    $$Gaussian(x, \mu, \sigma) = exp \Big( \frac{- (x - \mu)^2}{\sigma} \Big)$$
    2. **Height Transfer**: the current height of the vertex. Vertices that are on a higher elevation (on land) should be lowered more than vertices that have a lower elevation (in ocean). Therefore the height contribution will be calculated by passing the current heights of vertices through a sigmoid function. Given the centre $\mu$ and steepness $s$, our sigmoid is defined by:
    $$Sigmoid(x, \mu, s) = \frac{1}{1 + e^{- s (x - \mu)}}$$

    """

    def __init__(self):
        return

    def _sigmoid(self, x, centre=-0.1, steepness=6):
        """
        Sigmoid function used for height transfer of diverge lowering
        """
        return 1 / (1 + np.exp(-(x - centre) * steepness))

    def _setDistanceToDivergence(self):
        """
        This function gets the distance to divergence boundary used in the lowering tranfer function.
        """

        divXYZ, divLinePoints = [], []

        # Get coordinates and line points of all diverging plate boundary locations
        for bound in self.pBounds:
            for i in range(bound.lineCentres.shape[0]):
                if bound.collisionSpeed[i] < 0:
                    divXYZ.append(bound.lineCentres[i])
                    divLinePoints.append(bound.linePoints[i])

        # Get distance from vertices to diverging plate boundaries.
        divXYZ = np.array(divXYZ)
        linePoints = np.array(divLinePoints)
        distIds = cKDTree(divXYZ).query(self.xyz, k=1)[1]
        distIds[distIds >= divXYZ.shape[0]] = divXYZ.shape[0] - 1
        closestLinePoints = linePoints[distIds]
        self.distToDivs = _PlateBoundary.getDistsToLinesSeg(self.xyz, closestLinePoints)

        return

    def getDivergeLowering(
        self, gaussMean=0, gaussVariance=0.25, sigmoidCentre=-0.1, sigmoidSteepness=6
    ):
        """
        This function is the main entry point to compute subsidence along diverging plates.

        It is used to get the appropriate heights and distance.

        The heights input is read from the current mesh elevation. To get the distances from diverging plate boundaries, we first get the location of all diverging plate boundaries, which can be identified by having a negative collision speed. The distances will then be calculated based on the boundary lines.
        """

        self._setDistanceToDivergence()
        distanceTransfer = self._gaussian(
            self.distToDivs / self.maxLoweringDistance,
            mean=gaussMean,
            variance=gaussVariance,
        )
        heightTransfer = self._sigmoid(
            self.elev / self.minMaxLoweringHeights,
            centre=sigmoidCentre,
            steepness=sigmoidSteepness,
        )

        self.uplifts -= self.baseLowering * distanceTransfer * heightTransfer

        return
