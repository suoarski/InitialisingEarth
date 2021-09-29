from time import process_time

from .tools import ReadYaml as _ReadYaml
from .mesher import EarthSurf as _EarthSurf
from .tectonics import PlateInfo as _PlateInfo
from .tectonics import Uplift as _Uplift
from .tectonics import Divergence as _Divergence
from .mesher import EarthFcts as _EarthFcts
from .tools import WriteMesh as _WriteMesh


class Model(
    _ReadYaml, _WriteMesh, _EarthSurf, _PlateInfo, _Uplift, _Divergence, _EarthFcts,
):
    """
    Instantiates model object and reconstruct Earth evolution.

    This object contains methods for the following operations:

     - initialisation of parameters based on input file options
     - computation of surface evolution over time
     - computation of tectonic forcing based on plate velocities

    :arg filename: YAML input file
    :arg verbose: output flag for model main functions
    """

    def __init__(self, filename, verbose=True, *args, **kwargs):

        t0 = process_time()
        self.verbose = verbose

        # Read input dataset
        _ReadYaml.__init__(self, filename)

        # Initialise earth plates and functions
        _EarthFcts.__init__(self)
        _EarthSurf.__init__(self)
        _PlateInfo.__init__(self)
        _Uplift.__init__(self)
        _Divergence.__init__(self)

        # Initialise output mesh
        _WriteMesh.__init__(self)

        print(
            "\n--- Initialisation Phase (%0.02f seconds)" % (process_time() - t0),
            flush=True,
        )

        return

    def runStep(self):
        """
        Perform a single iteration of evolution. This function will first remesh the advected surface based on plate velocity and then will update the plate parameters and deduce the vertical displacements.
        """

        _EarthSurf.remeshSurface(self)
        # Advance time
        self.tNow -= self.dt
        _PlateInfo.updatePlates(self)
        _WriteMesh.visModel(self)

        return

    def runProcesses(self):
        """
        Main entry point to run the simulation over time. This function calls the `runStep` function described above.
        """

        while self.tNow > self.tEnd:
            print(
                "\n+++ Compute evolution from "
                + str(self.tNow)
                + "Ma to "
                + str(self.tNow - self.dt)
                + "Ma"
            )
            self.runStep()

        return
