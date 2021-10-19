import os
import numpy as np
import pandas as pd
import ruamel.yaml as yaml


class ReadYaml(object):
    """
    Class for reading simulation input file and initialising model parameters.

    TODO: definition of input parameters will have to be provided in the package documentation.
    """

    def __init__(self, filename):
        """
        Parsing YAML file.

        :arg filename: input filename (.yml YAML file)
        """

        # Check input file exists
        try:
            with open(filename) as finput:
                pass
        except IOError:
            print("Unable to open file: ", filename, flush=True)
            raise IOError("The input file is not found...")

        # Open YAML file
        with open(filename, "r") as finput:
            self.input = yaml.load(finput, Loader=yaml.Loader)

        if "name" in self.input.keys() and self.verbose:
            print(
                "The following model will be run:     {}".format(self.input["name"]),
                flush=True,
            )

        # Read simulation parameters
        self._readDomain()
        self._readTime()
        self._readDBSCAN()
        self._readTecto()
        self._readPaleo()
        self._readOut()

        self.radius = 6378137.0
        self.tNow = self.tStart
        self.saveTime = self.tNow

        if self.clustdist is None:
            error_msg = "If using a refinement level below 7 or above 10, the cluster distance parameter `clustdist` needs to be defined in the `dbscan` module."
            raise RuntimeError(error_msg)

        return

    def _readDomain(self):
        """
        Read domain definition, refinement level and other  model parameters.

        For specific refinement level the clustering distance used in the `dbscan` algorithm is automatically assigned.
        """

        try:
            domainDict = self.input["domain"]
        except KeyError:
            print(
                "Key 'domain' is required and is missing in the input file!", flush=True
            )
            raise KeyError("Key domain is required in the input file!")

        try:
            self.reflvl = domainDict["reflvl"]
        except KeyError:
            self.reflvl = 8

        if self.reflvl >= 7 and self.reflvl < 11:
            data = {
                "reflvl": np.arange(7, 11),
                "cluster_eps": [50.0e3, 25.0e3, 10.0e3, 5.0e3],
            }
            df = pd.DataFrame(data)
            self.clustdist = df.loc[df["reflvl"] == self.reflvl]["cluster_eps"].values[
                0
            ]

        try:
            self.interp = domainDict["interp"]
        except KeyError:
            self.interp = 1

        try:
            gospl = domainDict["gospl"]
            # Get output directory
            if gospl is not None:
                self.gospl = os.getcwd() + "/" + gospl
                if not os.path.exists(self.gospl):
                    os.makedirs(self.gospl)

        except KeyError:
            self.gospl = None

        return

    def _readTime(self):
        """
        Read simulation time declaration.
        """

        try:
            timeDict = self.input["time"]
        except KeyError:
            print(
                "Key 'time' is required and is missing in the input file!", flush=True
            )
            raise KeyError("Key time is required in the input file!")

        try:
            self.tStart = timeDict["start"]
        except KeyError:
            print(
                "Key 'start' is required and is missing in the 'time' declaration!",
                flush=True,
            )
            raise KeyError("Simulation start time needs to be declared.")

        try:
            self.tEnd = timeDict["end"]
        except KeyError:
            print(
                "Key 'end' is required and is missing in the 'time' declaration!",
                flush=True,
            )
            raise KeyError("Simulation end time needs to be declared.")

        if self.tStart <= self.tEnd:
            raise ValueError("Simulation end/start times do not make any sense!")

        try:
            self.dt = timeDict["dt"]
        except KeyError:
            print(
                "Key 'dt' is required and is missing in the 'time' declaration!",
                flush=True,
            )
            raise KeyError("Simulation discretisation time step needs to be declared.")

        return

    def _readDBSCAN(self):
        """
        Parse dbscan forcing conditions.
        """

        try:
            dbscanDict = self.input["dbscan"]
            try:
                self.nprocs = dbscanDict["nprocs"]
            except KeyError:
                self.nprocs = 1

            try:
                self.clustngbh = dbscanDict["clustngbh"]
            except KeyError:
                self.clustngbh = 6

            try:
                clustdist = dbscanDict["clustdist"]
            except KeyError:
                clustdist = self.clustdist

            if self.clustdist is None:
                self.clustdist = clustdist

        except KeyError:
            self.nprocs = 1
            self.clustngbh = 6

        return

    def _readTecto(self):
        """
        Parse tecto forcing condition paramters (for converging and diverging plates).
        """

        try:
            tectoDict = self.input["tecto"]
            try:
                self.baseUplift = tectoDict["buplift"] * 1.0e3
            except KeyError:
                self.baseUplift = 2 * 1.0e3

            try:
                self.distTransRange = tectoDict["transfer"] * 1.0e3
            except KeyError:
                self.distTransRange = 1000 * 1.0e3

            try:
                self.numToAverageOver = tectoDict["boundpts"]
            except KeyError:
                self.numToAverageOver = 10

            try:
                self.baseLowering = tectoDict["blower"] * 1.0e3
            except KeyError:
                self.baseLowering = 2 * 1.0e3

            try:
                self.minMaxLoweringHeights = tectoDict["mlower"] * 1.0e3
            except KeyError:
                self.minMaxLoweringHeights = 8 * 1.0e3

            try:
                self.maxLoweringDistance = tectoDict["dlower"] * 1.0e3
            except KeyError:
                self.maxLoweringDistance = 2000 * 1.0e3

        except KeyError:
            self.baseUplift = 2 * 1.0e3
            self.distTransRange = 1000 * 1.0e3
            self.numToAverageOver = 10
            self.baseLowering = 2 * 1.0e3
            self.minMaxLoweringHeights = 8 * 1.0e3
            self.maxLoweringDistance = 2000 * 1.0e3

        return

    def _readPaleo(self):
        """
        Parse paleodata forcing conditions, rotation file, plate polygon file, paleosurface folder and paleo plateID folder.
        """

        try:
            paleoDict = self.input["paleodata"]
            try:
                self.paleoDemsPath = paleoDict["dem"]
            except KeyError:
                print(
                    "Key 'dem' is required and is missing in the 'paleodata' declaration!",
                    flush=True,
                )
                raise KeyError("Dem is not defined!")

            try:
                self.paleoRainPath = paleoDict["rain"]
            except KeyError:
                self.paleoRainPath = None

            try:
                self.paleoDemForce = paleoDict["demforce"]
            except KeyError:
                self.paleoDemForce = False

            try:
                self.tecForce = paleoDict["tecforce"]
            except KeyError:
                self.tecForce = None

            if self.tecForce is not None and self.paleoDemForce:
                raise KeyError(
                    "Cannot have tecforce and demforce defined in the same run"
                )

            try:
                self.gaussval = paleoDict["demsmth"]
            except KeyError:
                self.gaussval = 0.0

            try:
                self.paleoVelocityPath = paleoDict["vel"]
            except KeyError:
                print(
                    "Key 'vel' is required and is missing in the 'paleodata' declaration!",
                    flush=True,
                )
                raise KeyError("Vel is not defined!")

            try:
                self.rotationsDirectory = paleoDict["rot"]
            except KeyError:
                print(
                    "Key 'rot' is required and is missing in the 'paleodata' declaration!",
                    flush=True,
                )
                raise KeyError("Rotation file is not defined!")

            try:
                self.platePolygonsDirectory = paleoDict["plate"]
            except KeyError:
                print(
                    "Key 'plate' is required and is missing in the 'paleodata' declaration!",
                    flush=True,
                )
                raise KeyError("Plate file is not defined!")
        except KeyError:
            print(
                "Key 'paleodata' is required and is missing in the input file!",
                flush=True,
            )
            raise KeyError("Key paleodata is required in the input file!")

        return

    def _readOut(self):
        """
        Parse output directory declaration.
        """

        try:
            outDict = self.input["output"]
            try:
                self.outputDir = outDict["dir"]
            except KeyError:
                self.outputDir = "output"
            try:
                self.makedir = outDict["makedir"]
            except KeyError:
                self.makedir = True
        except KeyError:
            self.outputDir = "output"
            self.makedir = True

        return
