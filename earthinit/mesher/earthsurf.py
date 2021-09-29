import numpy as np
import xarray as xr
import stripy as stripy
from pathlib import Path
from scipy import ndimage
from time import process_time


class EarthSurf(object):
    """
    This class encapsulates all the functions related to earth surface definition and evolution induced by plates movement. It is first called by the framework to initialise an initial paleosurface from an input netcdf file  and then remesh the surface over time using a clustering algorithm to estimate topography in subducting region.
    """

    def __init__(self):

        self.interpZ = None

        # Build the mesh
        t0 = process_time()
        self._buildMesh()
        if self.verbose:
            print(
                "\nBuilding icosahedral mesh (%0.02f seconds)" % (process_time() - t0),
                flush=True,
            )

        # Write gospl mesh properties to file
        if self.gospl is not None:
            self._writeGosplMesh(mesh=True)

        return

    def _buildMesh(self):
        """
        Build an icosahedral mesh using `stripy` and interpolate paleosurface elevation on it.

        The function defines the mesh spherical and lon/lat coordinates as well as the triangular cells
        that form the unstructured mesh.

        TODO: this function could be extended to actually work with other types of mesh
        """

        # Using stripy and the user defined refinement level construct the spherical mesh
        if self.reflvl < 11:
            grid = stripy.spherical_meshes.icosahedral_mesh(
                include_face_points=False, refinement_levels=self.reflvl
            )
        else:
            grid = stripy.spherical_meshes.octahedral_mesh(
                include_face_points=False, refinement_levels=self.reflvl
            )

        # Output the mesh characteristics
        if self.verbose:
            str_fmt = "{:25} {:9}"
            print(str_fmt.format("Number of points", grid.npoints))
            print(str_fmt.format("Number of cells", grid.simplices.shape[0]))
            edges = np.multiply(grid.edge_lengths(), self.radius)
            self.minlgth = edges.min()
            self.maxlgth = edges.max()

            str_fmt = "{:30} {:9}"
            print("")
            print(str_fmt.format("Minimum edge length (km)", int(self.minlgth / 1.0e3)))
            print(str_fmt.format("Maximum edge length (km)", int(self.maxlgth / 1.0e3)))
            print(
                str_fmt.format(
                    "Distance for clustering (km)", int(self.clustdist / 1.0e3)
                )
            )

        # Define mesh properties
        self.xyz = np.vstack((grid.points[:, 0], grid.points[:, 1]))
        self.xyz = np.vstack((self.xyz, grid.points[:, 2])).T
        self.xyz = np.multiply(self.xyz, self.radius)
        self.npoints = len(self.xyz)

        self.cells = grid.simplices.copy()
        self.ncells = len(self.cells)

        # Get a xarray data from paleosurface file
        paleoData = self._getPaleoTopo()

        # Convert spherical mesh longitudes and latitudes from radian to degree
        glat = np.mod(np.degrees(grid.lats) + 90, 180.0)
        glon = np.mod(np.degrees(grid.lons) + 180.0, 360.0)
        elev = paleoData.z.values.T

        # Map mesh coordinates on this dataset
        lon1 = elev.shape[0] * glon / 360.0
        lat1 = elev.shape[1] * glat / 180.0
        coord1 = np.stack((lon1, lat1))
        self.elev = ndimage.map_coordinates(
            elev, coord1, order=2, mode="nearest"
        ).astype(np.float64)

        # Mesh lon/lat coordinates
        meshlon, meshlat = self._cartesianToPolarCoords(self.xyz)
        self.lonlat = np.stack((meshlon, meshlat)).T

        return

    def _getPaleoTopo(self):
        """
        Earth paleosurface is read from a netcdf files containing lon, lat and elevation at a specific geological time interval.

        Such paleo-surfaces can be obtained from various places (e.g. the following [link](https://zenodo.org/record/5460860), or gPlates portal).

        Here, we use `xarray` to open the dataset.

        :return: a xarray dataset of paleosurface

        """

        # Get the paleosurface mesh file (as netcdf file)
        paleoDemsPath = Path(self.paleoDemsPath)
        initialLandscapePath = list(
            paleoDemsPath.glob("**/%dMa.nc" % int(self.tStart))
        )[0]

        # Open it with xarray
        data = xr.open_dataset(initialLandscapePath)
        lon_name = "longitude"
        data["_longitude_adjusted"] = xr.where(
            data[lon_name] < 0, data[lon_name] + 360, data[lon_name]
        )
        data = (
            data.swap_dims({lon_name: "_longitude_adjusted"})
            .sel(**{"_longitude_adjusted": sorted(data._longitude_adjusted)})
            .drop(lon_name)
        )
        data = data.rename({"_longitude_adjusted": lon_name})

        return data.sortby(data.latitude)

    def _moveSurface(self):
        """
        Move initial mesh according to each plate velocity field. To move the tectonic plates, we use `pygplates`.

        The function initialises a new Numpy array `nxyz` that contains the mesh coordinates after displacement.
        """

        self.nxyz = self.xyz.copy()
        for idx in np.unique(self.plateIds):
            rot = self.rotations[idx]
            self.nxyz[self.plateIds == idx] = rot.apply(self.nxyz[self.plateIds == idx])
            self.nxyz[self.plateIds == idx]

        return

    def _subductHeightTect(self, output=False, cwd="."):
        """
        This function set heights and vertical displacements of subducting vertices to the heights of nearby over-riding vertices.

        After displacement, vertices in subduction regions are closer to each other and are identified using the `dbscan` clustering algorithm. Then, for each vertex belonging to a cluster, we set its height and vertical tectonic regime to the maximum of its nearest neighbours.
        """
        self._dbscanMPI(output, cwd)

        # Get heights of nearest neighbours
        idCluster = self.isCluster > 0
        heightsInCluster = self.elev[idCluster]
        neighbourHeights = heightsInCluster[self.clustNgbhs]

        # For points in cluster, set new heights to the maximum height of
        # nearest neighbours
        self.clustZ = self.elev.copy()
        self.clustZ[idCluster] = np.max(neighbourHeights, axis=1)

        # Get tectonics of nearest neighbours
        idCluster = self.isCluster > 0
        tectoInCluster = self.uplifts[idCluster]
        neighbourTecto = tectoInCluster[self.clustNgbhs]

        # For points in cluster, set new heights to the maximum height of
        # nearest neighbours
        self.clustTec = self.uplifts.copy()
        self.clustTec[idCluster] = np.max(neighbourTecto, axis=1)

        return

    def remeshSurface(self):
        """
        This is the main function of this class and calls the following private functions:

        - _moveSurface
        - _subductHeightTect

        Once the mesh coordinates have been moved and the elevations and tectonic regime in subduction regions have been updated, the function interpolates the advected dataset to the initial mesh using a kdtree search and applying an inverse weighting distance function.

        """

        t0 = process_time()
        self._moveSurface()
        if self.verbose:
            print(
                "\nMove plates (%0.02f seconds)" % (process_time() - t0), flush=True,
            )

        t0 = process_time()
        self._subductHeightTect()

        # Build the kdtree
        self.distNbghs, self.idNbghs = self.ptree.query(self.xyz, k=self.interp)
        if self.interp == 1:
            self.interpZ = self.clustZ[self.idNbghs]
            self.interpT = self.clustTec[self.idNbghs]
        else:
            # Inverse weighting distance...
            weights = np.divide(
                1.0,
                self.distNbghs,
                out=np.zeros_like(self.distNbghs),
                where=self.distNbghs != 0,
            )
            onIDs = np.where(self.distNbghs[:, 0] == 0)[0]
            temp = np.sum(weights, axis=1)
            tmp = np.sum(weights * self.clustZ[self.idNbghs], axis=1)
            # Elevation
            self.interpZ = np.divide(
                tmp, temp, out=np.zeros_like(temp), where=temp != 0
            )
            tmp = np.sum(weights * self.clustTec[self.idNbghs], axis=1)
            # Vertical displacements
            self.interpT = np.divide(
                tmp, temp, out=np.zeros_like(temp), where=temp != 0
            )
            if len(onIDs) > 0:
                self.interpT[onIDs] = self.clustTec[self.idNbghs[onIDs, 0]]

        if self.verbose:
            print(
                "Update elevations and tectonics regime (%0.02f seconds)"
                % (process_time() - t0),
                flush=True,
            )

        # Write for considered time step the parameters used in a gospl model
        if self.gospl is not None:
            self._writeGosplMesh()

        return