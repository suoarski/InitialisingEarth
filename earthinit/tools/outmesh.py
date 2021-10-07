import os
import h5py
import glob
import shutil
import meshplex
import numpy as np
from time import process_time


class WriteMesh(object):
    """
    Class for writing model outputs. The outputs are written as hdf5 files.

    .. note::

        The model outputs are all located in an output folder and consist of a time series file named `earth.xdmf` and 2 other folders (`h5` and `xmf`).

    The `XDMF` file is the main entry point for visualising the output and should be sufficient for most users. This file can easely be opened within `Paraview <https://www.paraview.org/download/>`_.
    """

    def __init__(self):
        """
        Initialise model outputs paramters.
        """

        self.outstep = 0
        self.saveTime = self.tStart

        self._createOutputDir()

        self.visModel()

        return

    def visModel(self):
        """
        Main function to write model outputs on disk.
        """

        # Output time step for first step
        if self.saveTime == self.tStart:
            t0 = process_time()
            self._outputMesh()
            self.saveTime -= self.dt
            if self.verbose:
                print(
                    "Writing  outputs (%0.02f seconds)" % (process_time() - t0),
                    flush=True,
                )

        # Output time step after start time
        elif self.tNow >= self.saveTime:
            t0 = process_time()
            self._outputMesh()
            self.saveTime -= self.dt
            if self.verbose:
                print(
                    "Writing  outputs (%0.02f seconds)" % (process_time() - t0),
                    flush=True,
                )

        return

    def _writeGosplMesh(self, mesh=False):
        """
        Main function to write gospl mesh and tectonic forcing input files.

        Gospl will change the elevation by applying erosion and deposition over the prescribe surface. Here we save on the file the indices and neighbours from the clustering algorithm and kdtree used in the remeshing function.
        """

        from gospl._fortran import definegtin

        if mesh:
            gosplmesh = (
                self.gospl
                + "/mesh_"
                + str(self.reflvl)
                + "_"
                + str(int(self.tStart))
                + "Ma"
            )

            t0 = process_time()
            Gmesh = meshplex.MeshTri(self.xyz, self.cells)
            s = Gmesh.idx_hierarchy.shape
            a = np.sort(Gmesh.idx_hierarchy.reshape(s[0], -1).T)
            Gmesh.edges = {"points": np.unique(a, axis=0)}
            ngbNbs, ngbID = definegtin(
                len(self.xyz), Gmesh.cells("points"), Gmesh.edges["points"]
            )

            np.savez_compressed(
                gosplmesh,
                v=self.xyz,
                c=self.cells,
                n=ngbID[:, :8].astype(int),
                z=self.elev,
            )
        else:
            gosplmesh = (
                self.gospl
                + "/plate_"
                + str(self.reflvl)
                + "_"
                + str(int(self.tNow))
                + "Ma"
            )

            t0 = process_time()
            np.savez_compressed(
                gosplmesh,
                iplate=self.plateIds,
                clust=self.isCluster,
                cngbh=self.clustNgbhs,
                dngbh=self.distNbghs,
                ingbh=self.idNbghs,
            )

            gosplmesh = (
                self.gospl
                + "/tecto_"
                + str(self.reflvl)
                + "_"
                + str(int(self.tNow))
                + "Ma"
            )

            t0 = process_time()
            np.savez_compressed(
                gosplmesh, z=self.interpZ, t=self.interpT,
            )

            if self.paleoRainPath is not None:
                gosplmesh = (
                    self.gospl
                    + "/rain_"
                    + str(self.reflvl)
                    + "_"
                    + str(int(self.tNow))
                    + "Ma"
                )

                t0 = process_time()
                np.savez_compressed(
                    gosplmesh, r=self.rain,
                )

        if self.verbose:
            print(
                "Writing gospl mesh data (%0.02f seconds)" % (process_time() - t0),
                flush=True,
            )

        return

    def _createOutputDir(self):
        """
        Create a directory to store outputs. By default the folder will be called `output`. If a folder name is specified in the YAML input file, this name will be used.

        .. note::
            The input option `makedir` gives the ability to delete any existing output folder with the same name (if set to `False`) or to create a new folder with the given dir name plus a number at the end (*e.g.* `outputDir_XX` if set to `True` with `XX` the run number). It prevents overwriting on top of previous runs.

        """

        # Get output directory
        if self.outputDir is not None:
            self.outputDir = os.getcwd() + "/" + self.outputDir
        else:
            self.outputDir = os.getcwd() + "/output"

        if self.makedir:
            if os.path.exists(self.outputDir):
                self.outputDir += "_" + str(
                    len(glob.glob(self.outputDir + str("*"))) - 1
                )
        else:
            if os.path.exists(self.outputDir):
                shutil.rmtree(self.outputDir, ignore_errors=True)

        os.makedirs(self.outputDir)
        os.makedirs(self.outputDir + "/h5")
        os.makedirs(self.outputDir + "/xmf")

        return

    def _outputMesh(self):
        """
        Saves mesh local information stored in the icosahedral mesh to HDF5 file. If the file already exists, it will be overwritten. The following variables will be available:

        - surface elevation `elev`.
        - plate id `plateid`.
        - vertical tectonic `uplift`.
        - distance to plate boundary `dist`.

        """

        if self.outstep == 0:
            topology = self.outputDir + "/h5/topology.h5"
            with h5py.File(topology, "w") as f:
                f.create_dataset(
                    "coords",
                    shape=(self.npoints, 3),
                    dtype="float32",
                    compression="gzip",
                )
                f["coords"][:, :] = self.xyz
                f.create_dataset(
                    "cells", shape=(self.ncells, 3), dtype="int32", compression="gzip",
                )
                f["cells"][:, :] = self.cells + 1

        h5file = self.outputDir + "/h5/earth." + str(self.outstep) + ".h5"
        with h5py.File(h5file, "w") as f:
            f.create_dataset(
                "elev", shape=(self.npoints, 1), dtype="float32", compression="gzip",
            )
            f["elev"][:, 0] = self.elev
            f.create_dataset(
                "plateid", shape=(self.npoints, 1), dtype="int32", compression="gzip",
            )
            f["plateid"][:, 0] = self.plateIds

            f.create_dataset(
                "uplift", shape=(self.npoints, 1), dtype="float32", compression="gzip",
            )
            f["uplift"][:, 0] = self.interpT

            if self.paleoRainPath is not None:
                f.create_dataset(
                    "rain",
                    shape=(self.npoints, 1),
                    dtype="float32",
                    compression="gzip",
                )
                f["rain"][:, 0] = self.rain

            if not self.paleoDemForce:
                f.create_dataset(
                    "dist",
                    shape=(self.npoints, 1),
                    dtype="float32",
                    compression="gzip",
                )
                f["dist"][:, 0] = self.distToBound

        self._save_XMF()
        self._save_XDMF()

        self.outstep += 1

        return

    def _save_XMF(self):
        """
        Saves mesh local information stored in the HDF5 to XmF file. The XMF files are XML schema explaining how to read `gospl` data files.

        The XmF file is written by a single processor (rank 0) and contains each partition HDF5 files in blocks. The variables described for the HDF5 file (function `_outputMesh` above) are all accessible from this file.

        """

        xmf_file = self.outputDir + "/xmf/earth" + str(self.outstep) + ".xmf"

        f = open(xmf_file, "w")

        # Header for xml file
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
        f.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write(" <Domain>\n")
        f.write('    <Grid GridType="Collection" CollectionType="Spatial">\n')
        f.write(
            '      <Time Type="Single" Value="%0.02f"/>\n' % -(self.saveTime * 1.0e6)
        )

        pfile = "h5/earth." + str(self.outstep) + ".h5"
        tfile = "h5/topology.h5"

        f.write('      <Grid Name="Block.0">\n')
        f.write(
            '         <Topology Type="Triangle" NumberOfElements="%d" BaseOffset="1">\n'
            % self.ncells
        )
        f.write('          <DataItem Format="HDF" DataType="Int" ')
        f.write('Dimensions="%d 3">%s:/cells</DataItem>\n' % (self.ncells, tfile))
        f.write("         </Topology>\n")

        f.write('         <Geometry Type="XYZ">\n')
        f.write('          <DataItem Format="HDF" NumberType="Float" Precision="4" ')
        f.write('Dimensions="%d 3">%s:/coords</DataItem>\n' % (self.npoints, tfile))
        f.write("         </Geometry>\n")

        f.write('         <Attribute Type="Scalar" Center="Node" Name="Z">\n')
        f.write('          <DataItem Format="HDF" NumberType="Float" Precision="4" ')
        f.write('Dimensions="%d 1">%s:/elev</DataItem>\n' % (self.npoints, pfile))
        f.write("         </Attribute>\n")

        f.write('         <Attribute Type="Scalar" Center="Node" Name="plateid">\n')
        f.write('          <DataItem Format="HDF" NumberType="Int" Precision="4" ')
        f.write('Dimensions="%d 1">%s:/plateid</DataItem>\n' % (self.npoints, pfile))
        f.write("         </Attribute>\n")

        f.write('         <Attribute Type="Scalar" Center="Node" Name="vtec">\n')
        f.write('          <DataItem Format="HDF" NumberType="Float" Precision="4" ')
        f.write('Dimensions="%d 1">%s:/uplift</DataItem>\n' % (self.npoints, pfile))
        f.write("         </Attribute>\n")

        if self.paleoRainPath is not None:
            f.write('         <Attribute Type="Scalar" Center="Node" Name="rain">\n')
            f.write(
                '          <DataItem Format="HDF" NumberType="Float" Precision="4" '
            )
            f.write('Dimensions="%d 1">%s:/rain</DataItem>\n' % (self.npoints, pfile))
            f.write("         </Attribute>\n")

        if not self.paleoDemForce:
            f.write('         <Attribute Type="Scalar" Center="Node" Name="dist">\n')
            f.write(
                '          <DataItem Format="HDF" NumberType="Float" Precision="4" '
            )
            f.write('Dimensions="%d 1">%s:/dist</DataItem>\n' % (self.npoints, pfile))
            f.write("         </Attribute>\n")

        f.write("      </Grid>\n")
        f.write("    </Grid>\n")
        f.write(" </Domain>\n")
        f.write("</Xdmf>\n")
        f.close()

        return

    def _save_XDMF(self):
        """
        This function writes the XDmF file which is calling the XmF files above. The XDmF file represents the *time series* of the model outputs and can be directly loaded and visualised with `Paraview <https://www.paraview.org/download/>`_.

        .. note::

            For a brief overview of the approach used to record `gospl` outputs, user can read this `visit documentation <https://www.visitusers.org/index.php?title=Using_XDMF_to_read_HDF5>`_
        """

        xdmf_file = self.outputDir + "/earth.xdmf"
        f = open(xdmf_file, "w")

        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
        f.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write(" <Domain>\n")
        f.write('    <Grid GridType="Collection" CollectionType="Temporal">\n')

        for s in range(self.outstep + 1):
            xmf_file = "xmf/earth" + str(s) + ".xmf"
            f.write(
                '      <xi:include href="%s" xpointer="xpointer(//Xdmf/Domain/Grid)"/>\n'
                % xmf_file
            )

        f.write("    </Grid>\n")
        f.write(" </Domain>\n")
        f.write("</Xdmf>\n")
        f.close()

        return
