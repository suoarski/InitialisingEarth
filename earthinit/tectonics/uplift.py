import numpy as np
from numba import jit
from scipy import special
from scipy.spatial import cKDTree

from .boundary import PlateBoundary as _PlateBoundary


class Uplift(object):
    r"""
    When two tectonic plates collide, one will typically submerge underneath the other, pushing the overriding plate upwards which leads to mountain growth. The class `SubductionBoundary` contains lists of overriding and subducting plates IDs which can be used to identify to apply subduction uplift on overriding plates.

    This class encapsulates the functions require to approximate the uplift $\delta h$ on the overriding nodes. Mainly these functions consider:

    - $f(d, \theta)$: The distance $d$ an overriding vertex is from the subduction boundary, and the angle $\theta$ that a vertex makes with the direction of collision.
    - $g(s, \theta)$: The speed $s$ at which nearby subduction boundaries are colliding with, and the angle $\theta$.
    - $h(r_h)$: The current height $r_h$ of the vertex

    .. note::

        We treat orogenic belts (continental collision) as a subduction boundary where both sides of the boundary are considered to be overriding plates.

    """

    def __init__(self):
        self.interpT = np.zeros(self.npoints)
        return

    def _gaussian(self, x, mean=0, variance=0.25):
        """
        Cumulative gaussian density function used in transfer function calculation.
        """

        return np.exp(-((x - mean) ** 2) / variance)

    # Skewed gaussian density function
    def _skewedGaussian(self, x, alpha=8, variance=0.25):
        r"""
        When creating a distance transfer function, we keep the main features of the transfer  between 0 and 1 so to specify the effective range $r$ of the distance transfer using $f(\frac{d}{r})$. The distance transfer function $f(x)$ is defined given a skewness $\alpha$:
        $$f(x) = \phi(x) \Phi(\alpha x)$$
        where $\phi(x, \mu, \sigma)$ is the standard gaussian density function with centre $\mu$ and variance $\sigma$ given by:
        $$\phi(x) = \exp \Big( \frac{- (x - \mu)^2}{\sigma} \Big)$$
        and $\Phi(x)$ is the cumulative gaussian density function given by:
        $$\Phi(x) = 1 + erf(x) = 1 + \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} dt$$
        where $erf(x)$ is the error function which is provided to us by the `scipy.special` library.
        """

        skewedGauss = self._gaussian(x, variance=variance) * (
            1 + special.erf(alpha * x)
        )
        return skewedGauss

    @staticmethod
    def getSpeedsData(subBounds):
        """
        Given a list of subduction boundaries, this function extract relevant data to compute the speed transfer.

        The approach for applying the speed transfer is:

        - For each vertex on a given plate, the closest valid subduction boundary is found.
        - Set the speed of each vertex to that of the closest subduction boundary

        .. note::

            Calculating speeds in this way results in 'jumps' in the speed transfer. One region of a plate might have a single common closest subduction boundary, and a neighbouring region might have a common closest subduction boundary with a completely different speeds. At the transition of these two regions, the speed transfer will form a 'jump' in speed, which isn't realistic. To avoid these sudden jumps in speed transfers, we will find the closest $N$ subduction boundary coordinates, and our speed transfers will be based on an average of the collisions speeds of those $N$ boundaries.

        """
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
        return (
            np.array(boundXYZ),
            boundSpeed,
            np.array(boundDirection),
            linePoints,
            lineLengths,
        )

    @staticmethod
    @jit(nopython=True)
    def getDirectionToBound(XYZ, closestLinePoints):
        """
        This function defines the direction to a specific boundary based on plate coordinates and closest boundary line segments.

        """

        directionToBound = np.zeros((XYZ.shape[0], closestLinePoints.shape[1], 3))
        for i in range(XYZ.shape[0]):
            linePoints = closestLinePoints[i]
            xyz = XYZ[i]

            # Get v and w to identify which case we are in
            v = linePoints[:, 1] - linePoints[:, 0]
            w = xyz - linePoints[:, 0]

            # Loop through each point to average over (later on)
            for j in range(w.shape[0]):
                # Case 1
                if np.dot(w[j], v[j]) <= 0:
                    direction = linePoints[j, 0] - xyz
                    directionToBound[i, j] = direction / np.linalg.norm(direction)
                # Case 2
                elif np.dot(v[j], v[j]) <= np.dot(w[j], v[j]):
                    direction = linePoints[j, 1] - xyz
                    directionToBound[i, j] = direction / np.linalg.norm(direction)
                # Case 3
                else:
                    # direction = np.cross(xyz, linePoints[j, 1] - linePoints[j, 0])
                    direction = np.cross(
                        linePoints[j, 0], linePoints[j, 1] - linePoints[j, 0]
                    )
                    directionToBound[i, j] = direction / np.linalg.norm(direction)

        return directionToBound

    def _getTransfersForThisPlate(self, subBounds, xyzOnThisPlate):
        """
        For a considered plate, this function calculates the corresponding transfer data.

        The angle transfer $\cos(\theta)$ is calculated by taking the dot product between the direction from point to boundary, and direction of this boundary's collision.

        .. note::

            Since the shape of our input arrays are not suitable for `np.dot()`, we  use `np.einsum()`. Since both direction arrays are already normalized, the dot product give the appropriate angle.

        """

        (
            boundXYZ,
            boundSpeed,
            bCollisionsDirection,
            linePoints,
            lineLengths,
        ) = self.getSpeedsData(subBounds)

        numToAverageOver = self.numToAverageOver

        # Get distances and the index array distIds
        distIds = cKDTree(boundXYZ).query(xyzOnThisPlate, k=numToAverageOver)[1]
        distIds[distIds >= boundXYZ.shape[0]] = (
            boundXYZ.shape[0] - 1
        )  # Avoid 'index out of bounds error'
        closestLinePoints = linePoints[distIds]
        distToBound = _PlateBoundary.getDistsToLinesSeg(
            xyzOnThisPlate, closestLinePoints[:, 0]
        )
        directionToBound = self.getDirectionToBound(xyzOnThisPlate, closestLinePoints)

        boundSpds = boundSpeed[distIds]
        # lineLngths = lineLengths[distIds]
        boundDirs = bCollisionsDirection[distIds]

        # Main transfer calculations
        speedTrans = np.zeros((numToAverageOver, xyzOnThisPlate.shape[0]))
        angleTrans = np.zeros((numToAverageOver, xyzOnThisPlate.shape[0]))
        for j in range(numToAverageOver):
            cosAngle = np.einsum(
                "ij,ij->i", directionToBound[:, j], boundDirs[:, j]
            )  # Dot product
            speedTrans[j] = np.abs(boundSpds[:, j] * cosAngle)
            angleTrans[j] = np.abs(cosAngle)
        return speedTrans, angleTrans, distToBound

    # Calculates transfers for all plates on the sphere
    def _getUpliftTransfers(self):
        r"""
        The speed transfers for each nearby subduction boundary, is calculated by:
        $$ g(s, \theta) = \| s \cos(\theta) \| $$
        where $s$ is the speed of boundary collision. The purpose of including $\cos(\theta)$ is to ensure that points on a plate in the direction of collision are more heavily influenced.
        The distance transfer is finally obtained by including the angle transfer $\cos(\theta)$ in its calculation. This is done by passing the effective distance $d_{eff}$ into the distance transfer function discussed previously:
        $$f(d_{eff}) =  f \Big( \frac{d_a}{\alpha \cos(\theta)} \Big)$$
        where $d_a$ is the actual distance. This change will have make the distance transfer have a further range in the direction of collision.

        """

        speedTransfer = np.zeros(self.npoints)
        angleTransfer = np.zeros(self.npoints)
        distToSubBounds = np.ones(self.npoints) * 1000

        for idx in np.unique(self.plateIds):
            if idx not in self.idToSubBound.keys():
                continue

            subBounds = self.idToSubBound[idx]
            xyzOnThisPlate = self.xyz[self.plateIds == idx]
            speedTrans, angleTrans, distToBound = self._getTransfersForThisPlate(
                subBounds, xyzOnThisPlate
            )
            speedTransfer[self.plateIds == idx] = (
                np.sum(speedTrans, axis=0) / self.numToAverageOver
            )
            angleTransfer[self.plateIds == idx] = (
                np.sum(angleTrans, axis=0) / self.numToAverageOver
            )
            distToSubBounds[self.plateIds == idx] = distToBound

        # Calculate the distance transfer
        dTransInput = distToSubBounds / (self.distTransRange * angleTransfer + 0.01)
        distTransfer = self._skewedGaussian(dTransInput)

        return speedTransfer, distTransfer

    # Get subduction uplifts using speed and distance transfers.
    def getUplifts(self):
        """
        Main entry function to compute the uplift on overridding plates based on the functions described above.
        """

        self.speedTransfer, distTransfer = self._getUpliftTransfers()
        self.uplifts = distTransfer * self.speedTransfer
        self.uplifts[self.uplifts <= 0.0001] = 0.0001
        self.uplifts /= np.max(self.uplifts)
        self.uplifts *= self.baseUplift

        return
