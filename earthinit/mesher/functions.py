import struct
import subprocess
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


class EarthFcts(object):
    """
    This class encapsulates a series of functions used to perform coordinate transformations and to run sub process commands such as the parallel `dbscan` algorithm.
    """

    def __init__(self):
        return

    def _cartesianToPolarCoords(self, XYZ, useLonLat=True):
        """
        Coordinate transformation from cartesian to polar.

        :arg XYZ: spherical coordinates
        :arg useLonLat: boolean set to True when lon/lat coordinates are returned.

        :return: lon, lat when useLonLat is True
        :return: R, theta, phi (polar) when useLonLat is False

        """

        X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
        R = (X ** 2 + Y ** 2 + Z ** 2) ** 0.5
        theta = np.arctan2(Y, X)
        phi = np.arccos(Z / R)

        if useLonLat is True:
            theta, phi = np.degrees(theta), np.degrees(phi)
            lon, lat = theta - 180, 90 - phi
            lon[lon < -180] = lon[lon < -180] + 360
            return lon, lat
        else:
            return R, theta, phi

    def _polarToCartesian(self, radius, theta, phi, useLonLat=True):
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

    def _quaternion(self, axis, angle):
        """
        To move tectonic plates, we create a rotation quaternion for each plate, and apply rotations to all vertices based on their plate ids.

        :arg axis: axis used for the quaternion
        :arg angle: angle of rotation

        :return: rotation quaternion

        """
        return [
            np.sin(angle / 2) * axis[0],
            np.sin(angle / 2) * axis[1],
            np.sin(angle / 2) * axis[2],
            np.cos(angle / 2),
        ]

    def _runSubProcess(self, args, output=True, cwd="."):
        """


        :arg data: local elevation numpy array
        :arg k_neighbors: number of nodes to use when querying the kd-tree


        :return: newH updated sedimentary layer thicknesses after compaction

        Parameters
        ----------


        args : list of str, optional
            MPI execute command and any additional MPI arguments to pass,
            e.g. ['mpiexec', '-n', '8'].
        output : bool, optional
            Capture OpenMC output from standard out
        cwd : str, optional
            Path to working directory to run in. Defaults to the current working
            directory.

        Raises
        ------

        RuntimeError
            If the `dbscan` executable returns a non-zero status
        """

        # Launch a subprocess
        p = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Capture and re-print OpenMC output in real-time
        lines = []
        while True:
            # If OpenMC is finished, break loop
            line = p.stdout.readline()
            if not line and p.poll() is not None:
                break

            lines.append(line)
            if output:
                # If user requested output, print to screen
                print(line, end="")

        # Raise an exception if return status is non-zero
        if p.returncode != 0:
            # Get error message from output and simplify whitespace
            output = "".join(lines)
            if "ERROR: " in output:
                _, _, error_msg = output.partition("ERROR: ")
            elif "what()" in output:
                _, _, error_msg = output.partition("what(): ")
            else:
                error_msg = "dbscan aborted unexpectedly."
            error_msg = " ".join(error_msg.split())

            raise RuntimeError(error_msg)

    def _dbscanMPI(self, output=False, cwd="."):
        """
        This funciton performs the `dbscan` clustering algorithm in parallel based on the *Disjoint-Set Data Structure based Parallel DBSCAN clustering implementation (MPI version)* by Md. Mostofa Ali Patwary.

        The code is available from the following [repository](https://github.com/pedro-ricardo/Parallel-DBSCAN) and takes as input a binary file written in a single column with number of points (4 bytes N), number of dimensions (4 bytes) followed by the points coordinates (N x D floating point numbers).

        It produces an output as a netCDF file containing the coodinates named as columns (position_col_X1, position_col_X2, ...) and one additional column named cluster_id for the corresponding cluster id the point belong to.

        The function below first creates the required binary file based on advected mesh points, then performs the `dbscan` algorithm in parallel and finally reads the output netcdf and finds the clustered points using a kdtree search.

        """

        if output:
            print("\ndbscan MPI")
        # Create binary file for dbscan
        dims = [len(self.nxyz), 3]
        linepts = self.nxyz.ravel()
        lgth = len(linepts)

        fbin = "nodes" + str(self.tNow) + ".bin"
        with open(fbin, mode="wb") as f:
            f.write(struct.pack("i" * 2, *[int(i) for i in dims]))
            f.write(struct.pack("f" * (lgth), *[float(i) for i in linepts]))

        fnc = "clusters" + str(self.tNow) + ".nc"
        mpi_args = [
            "mpirun",
            "-np",
            str(self.nprocs),
            "dbscan",
            "-i",
            fbin,
            "-b",
            "-m",
            "2",
            "-e",
            str(self.clustdist),
            "-o",
            fnc,
        ]

        self._runSubProcess(mpi_args, output, cwd)

        if output:
            print("\nGet global ID of clustered vertices")

        # Open clustered node file
        cluster = xr.open_dataset(fnc)
        isClust = cluster.cluster_id.values > 0
        clustPtsX = cluster.position_col_X0.values[isClust]
        clustPtsY = cluster.position_col_X1.values[isClust]
        clustPtsZ = cluster.position_col_X2.values[isClust]
        clustPts = np.vstack((clustPtsX, clustPtsY))
        clustPts = np.vstack((clustPts, clustPtsZ)).T

        # Get clustered global ids
        self.ptree = cKDTree(self.nxyz)
        dist, ids = self.ptree.query(clustPts, k=1)
        self.isCluster = np.zeros(len(self.nxyz), dtype=int)
        self.isCluster[ids] = 1

        # Create KDTree to find nearest neighbours of each point in cluster
        idCluster = self.isCluster > 0
        ptsCluster = self.nxyz[idCluster]
        ctree = cKDTree(ptsCluster)
        _, clustNgbhs = ctree.query(ptsCluster, k=self.clustngbh)

        self.clustNgbhs = clustNgbhs[:, 1:]
        args = [
            "rm",
            fbin,
            fnc,
        ]
        self._runSubProcess(args, output, cwd)

        return
