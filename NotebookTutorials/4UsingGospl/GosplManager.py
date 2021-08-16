
import os
import h5py
import stripy
import meshplex
import numpy as np
import pyvista as pv
from pathlib import Path
import mergeBack as merger
from scipy import interpolate
from gospl.model import Model as sim
from gospl._fortran import definegtin
from EarthsAssistant import EarthAssist

#Class for seting up files for and running Gospl
class GosplManager:
    def __init__(self, earth,
                subdivisions=5,
                 mainOutputDirectory = 'GosplRuns',
                 newBackwardSimulationEveryNSteps = 5
                ):
        
        #Store attributes passed by class initiation and directory specifications
        self.earth = earth
        self.subdivisions = subdivisions
        self.mainOutputDirectory = mainOutputDirectory
        self.forwardOutputFolder = '/GosplOutputFiles'
        self.backwardOutputFolder = '/BackwardsOutput'
        self.ymlBackwardDirectories = []
        
        #Icosphere for the creation of npz files
        self.icosphereXYZ, self.icoCells = self.createIcosphere(subdivisions=subdivisions, radius=earth.earthRadius)
        self.newBackwardSimulationEveryNSteps = newBackwardSimulationEveryNSteps
        self.outputBackwardsDirectories = []
        
        self.backwardTimeSteps = np.arange(self.earth.endTime, self.earth.startTime+1, self.newBackwardSimulationEveryNSteps)
        
        #============================== Default YML Attributes ======================================================
        #These attributes can be changed directly after initializing a GosplManager instance
        
        #Name attribute
        self.name = 'Automatically generated YML file'
        
        #Domain attributes
        #self.npdata = dataDirectory + '/Elevations{}Mya'.format(subdivisions, earth.startTime)
        self.npdataFormat = '{}/Elevations{}Mya'
        self.npdataBackFormat = '{}/BackwardsElevations{}Mya'
        self.flowdir = 5
        self.interp = 1
        
        #Time attributes
        self.gosplStepsAtEachIteration = 2
        self.start = earth.startTime * 1000000
        self.end = earth.endTime * 1000000
        self.tout = earth.deltaTime * 1000000
        self.dt = earth.deltaTime * 1000000 / self.gosplStepsAtEachIteration
        self.tec = earth.deltaTime * 1000000
        
        #Stream Power Law attributes (SPL)
        self.wfill = 100.
        self.sfill = 10.
        self.K = 3e-8
        self.d = 0.42
        
        #Diffusion attributes
        self.hillslopeKa = 0.02
        self.hillslopeKm = 0.2
        self.dstep = 5
        self.sedK = 100.
        self.sedKf = 200.
        self.sedKw = 300
        
        #Sea level attributes
        self.position = 0.
        
        #Climate attributes
        self.uniform = 2.
        
        #ForcePaleo attributes
        self.steps = [int(earth.startTime), int(earth.endTime)]#, int(earth.deltaTime)]
        
        #Output Attributes
        self.dirFormat = 'GosplOutput{}'
        self.makedir = False
    
    #After creating a GosplManager object, this function will do everything to run a simulation
    #Alternatively you can call these functions individually
    def createAllFilesAndRunSimulation(self):
        self.makeDirectories()
        self.createDomainNPdataFile()
        self.createTectonicDispNPZfiles()
        self.runGosplSimulation()
        
    #============================== YML File Creation ======================================================
    #We generate a string containing all the content of the backwards and forwards YML file
    #For each section of the YML file, we have a seperate function creating a string for it,
    #We then combine all these sections into a single string and create the YML file
    def getNameString(self):
        return "\nname: {}\n\n".format(self.name)
    
    def getDomainString(self, time, backwards=False):
        domainFormat = "domain:\n  npdata: '{}'\n  flowdir: {}\n  fast: {}\n  backward: {}\n  interp: {}\n\n"
        npdataFormat = self.npdataFormat
        npdata = npdataFormat.format(self.npzFilesDirectory, time)
        domainString = domainFormat.format(
            npdata,
            self.flowdir,
            backwards,
            backwards,
            self.interp)
        return domainString
    
    def getTimeString(self, start, end):
        timeFormat = "time:\n  start: -{}\n  end: {}\n  tout: {}\n  dt: {}\n  tec: {}\n\n"
        self.dt = self.earth.deltaTime * 1000000 / self.gosplStepsAtEachIteration
        timeString = timeFormat.format(
            float(start),
            float(end),
            float(self.tout),
            float(self.dt),
            float(self.tec))
        return timeString
    
    def getSPLstring(self):
        splFormat = "spl:\n  wfill: {}\n  sfill: {}\n  K: {}\n  d: {}\n\n"
        splString = splFormat.format(
            self.wfill,
            self.sfill,
            self.getYMLscientificNotation(self.K),
            self.d)
        return splString
    
    def getDiffusionString(self):
        diffusionFormat = "diffusion:\n  hillslopeKa: {}\n  hillslopeKm: {}\n  dstep: {}\n  sedK: {}.\n  sedKf: {}.\n  sedKw: {}\n\n"
        diffusionString = diffusionFormat.format(
            self.hillslopeKa,
            self.hillslopeKm,
            self.dstep,
            int(self.sedK),
            int(self.sedKf),
            self.sedKw)
        return diffusionString
    
    def getSeaString(self):
        seaFormat = "sea:\n  position: {}\n\n"
        seaString = seaFormat.format(self.position)
        return seaString
    
    def getForwardTectonicString(self):
        tectonicString = "tectonic:\n"
        tectonicPartFormat = " - start: -{}.\n   end: -{}.\n   mapH: '{}/ForceSubdivisions{}Time{}Mya'\n"
        for time in self.earth.simulationTimes[:-1]:
            tectonicPart = tectonicPartFormat.format(
                1000000 * time, 
                1000000 * (time - self.earth.deltaTime),
                self.npzFilesDirectory,
                self.subdivisions,
                time)
            tectonicString += tectonicPart
        return tectonicString + '\n'
    
    def getBackwardsTectonicString(self):
        tectonicString = "tectonic:\n"
        tectonicPartFormat = " - start: -{}.\n   end: -{}.\n   mapH: '{}/BackwardsForceSubdivisions{}Time{}Mya'\n"
        times = self.earth.simulationTimes[:-1]
        for time in times:
            tectonicPart = tectonicPartFormat.format(
                1000000 * time, 
                1000000 * (time - self.earth.deltaTime),
                self.npzFilesDirectory,
                self.subdivisions,
                np.max(times) - time + self.earth.deltaTime)
            tectonicString += tectonicPart
        return tectonicString + '\n'
    
    def getClimateString(self):
        climateFormat = "  - start: -{}.\n    uniform: {}\n"
        times = self.backwardTimeSteps
        climateString = "climate:\n"
        for t in reversed(times):
            climateString += climateFormat.format(
                1000000 * t,
                self.uniform)
        return climateString + '\n'
    
    def getForcePaleoString(self):
        forcePaleoFormat = "forcepaleo:\n  dir: '{}'\n  steps: {}\n\n"
        forcePaleoString = forcePaleoFormat.format(
            self.thisRunDirectory + self.backwardOutputFolder,
            [self.newBackwardSimulationEveryNSteps for i in range(len(self.backwardTimeSteps) - 1)])
        return forcePaleoString
    
    def getOutputString(self, backwards=False, tme=0):
        outputFormat = "output:\n  dir: '{}'\n  makedir: {}\n\n"
        outputFolder = self.forwardOutputFolder
        if backwards:
            outputFolder = self.backwardOutputFolder + '{}Mya'.format(tme)
            self.outputBackwardsDirectories.append(self.thisRunDirectory + outputFolder)
        outputString = outputFormat.format(
            self.thisRunDirectory + outputFolder,
            self.makedir)
        return outputString
    
    #Create a string containing all the content of the YML file
    def getForwardYMLstring(self):
        name = self.getNameString()
        domain = self.getDomainString(self.earth.startTime)
        time = self.getTimeString(self.start, self.end)
        spl = self.getSPLstring()
        diffusion = self.getDiffusionString()
        sea = self.getSeaString()
        tectonic = self.getForwardTectonicString()
        climate = self.getClimateString()
        forcePaleo = self.getForcePaleoString()
        output = self.getOutputString()
        return name + domain + time + spl + diffusion + sea + tectonic + climate + forcePaleo + output
    
    #Create a string containing all the content of the YML file
    def getBackwardYMLstring(self, tme):
        name = self.getNameString()
        domain = self.getDomainString(self.earth.startTime - tme, backwards=True)
        time = self.getTimeString(1000000 * tme, -1000000 * (tme - self.newBackwardSimulationEveryNSteps))
        spl = self.getSPLstring()
        diffusion = self.getDiffusionString()
        sea = self.getSeaString()
        tectonic = self.getBackwardsTectonicString()
        climate = self.getClimateString()
        output = self.getOutputString(backwards=True, tme=tme)
        return name + domain + time + spl + diffusion + sea + tectonic + climate + output
    
    #If x is given in scientific notation, return x as string scientific notation compatible with YML files.
    @staticmethod
    def getYMLscientificNotation(x):
        if 'e' in str(x):
            x = str(x).split('e-')
            x = '{}.e-{}'.format(x[0], x[1])
        return x
    #============================== Create Directories and YML Files ==============================================
    def makeDirectories(self):
        
        #Create the main output directory if it does not already exist
        if not os.path.isdir(self.mainOutputDirectory):
            os.mkdir('./{}'.format(self.mainOutputDirectory))
        
        #Create subdirectory for this particular Gospl run
        runNumber = 1
        thisRunDirectory = './{}/run{}'.format(self.mainOutputDirectory, runNumber)
        while os.path.isdir(thisRunDirectory):
            runNumber += 1
            thisRunDirectory = './{}/run{}'.format(self.mainOutputDirectory, runNumber)
        self.thisRunDirectory = thisRunDirectory
        os.mkdir(thisRunDirectory)
        
        #Create directory for npz files
        self.npzFilesDirectory = thisRunDirectory + '/NPZfiles'
        os.mkdir(self.npzFilesDirectory)
        
        #Create the forward YML file
        ymlForwardContent = self.getForwardYMLstring()
        self.ymlForwardDirectory = thisRunDirectory + '/forward.yml'
        ymlForward = open(self.ymlForwardDirectory, 'w')
        ymlForward.write(ymlForwardContent)
        ymlForward.close()
        
        #Create a new YML file for each backwards simulation step
        createBackYmlAtTheseTimes = self.backwardTimeSteps
        for t in createBackYmlAtTheseTimes[1:]:
            ymlBackwardContent = self.getBackwardYMLstring(t)
            self.ymlBackwardDirectories.append(thisRunDirectory + '/Backward{}Mya.yml'.format(t))
            ymlBackward = open(self.ymlBackwardDirectories[-1], 'w')
            ymlBackward.write(ymlBackwardContent)
            ymlBackward.close()
    
    #============================== Create Domain Elevations NPdata ======================================================
    #Create an icosphere
    @staticmethod
    def createIcosphere(subdivisions=6, radius=6378137):
        icosphere = stripy.spherical_meshes.icosahedral_mesh( 
                        refinement_levels = subdivisions,
                        include_face_points = False)
        icosphereXYZ = icosphere._points * radius
        icoCells = icosphere.simplices
        return icosphereXYZ, icoCells
    
    #Create an icosphere and interpolate earth heights onto it
    def getIcoHeights(self, iteration=-1):
        earthHeights = self.earth.heightHistory[iteration]
        radLonLat = EarthAssist.cartesianToPolarCoords(self.icosphereXYZ)
        icoLonLat = np.stack((radLonLat[1], radLonLat[2]), axis=1)
        icoHeights = interpolate.griddata(self.earth.lonLat, earthHeights, icoLonLat, method='cubic')
        icoHeights = icoHeights[:, np.newaxis]
        return icoHeights
    
    #Create list of neighbour ids based on bfModel notebook tutorial
    def getNeighbourIds(self):
        Gmesh = meshplex.MeshTri(self.icosphereXYZ, self.icoCells)
        s = Gmesh.idx_hierarchy.shape
        a = np.sort(Gmesh.idx_hierarchy.reshape(s[0], -1).T)
        Gmesh.edges = {'points': np.unique(a, axis=0)}
        ngbNbs, ngbID = definegtin(len(self.icosphereXYZ), Gmesh.cells['points'], Gmesh.edges['points'])
        ngbIDs = ngbID[:,:8].astype(int)
        return ngbIDs
    
    def createDomainNPdataFiles(self):
        createNPDataAtTheseTimes = self.backwardTimeSteps
        for t in createNPDataAtTheseTimes:
            heights = self.getIcoHeights(iteration=t)
            neighbours = self.getNeighbourIds()
            fileName = self.npdataFormat.format(self.npzFilesDirectory, t)
            np.savez_compressed(fileName, v=self.icosphereXYZ, c=self.icoCells, n=neighbours.astype(int), z=heights)
            
    #============================== Create Tectonic Displacements Files =============================================
    #We interpolate the force field onto an Icosphere suitable for Gospl
    def interpolateForces(self, earthLonLat, icosphereXYZ, forceXYZ):
        radLonLat = EarthAssist.cartesianToPolarCoords(icosphereXYZ)
        icoLonLat = np.stack((radLonLat[1], radLonLat[2]), axis=1)
        icoForce = interpolate.griddata(earthLonLat, forceXYZ, icoLonLat, method='cubic')
        return icoForce
    
    #Create tectonic force displacement files
    def createTectonicDispNPZfiles(self):
        times = self.earth.simulationTimes[:-1]
        for i, time in enumerate(times):
            tectonicDisp = self.earth.tectonicDispHistory[i]
            icoForce = self.interpolateForces(self.earth.lonLat, self.icosphereXYZ, tectonicDisp)
            
            #Forward displacement files
            fileName = '{}/ForceSubdivisions{}Time{}Mya'.format(self.npzFilesDirectory, self.subdivisions, time)
            np.savez_compressed(fileName, xyz=icoForce)
            
            #Backward displacement files
            fileName = '{}/BackwardsForceSubdivisions{}Time{}Mya'.format(self.npzFilesDirectory, self.subdivisions, time)
            np.savez_compressed(fileName, xyz=-icoForce)
    
    #Run the gospl simulation
    def runGosplSimulation(self):
        
        #Run backward model
        for backYml in reversed(self.ymlBackwardDirectories):
            mod = sim(backYml, False, False)
            mod.runProcesses()
            mod.destroy()
        
        #Merge the output of the multiple backwards simulations into a single folder
        merger.mergeBackModels(self.ymlBackwardDirectories, self.thisRunDirectory + self.backwardOutputFolder)
        
        #Run forward model
        mod = sim(self.ymlForwardDirectory, False, False)
        mod.runProcesses()
        mod.destroy()
        
    #============================== View or Animate Gospl Results =============================================
    #Using the gospl data, we create a pyvista mesh
    def createGosplDataMesh(self, iteration, runNumber=None):
        if (runNumber != None):
            self.thisRunDirectory = './{}/run{}'.format(self.mainOutputDirectory, runNumber)
        gosplFilenamePattern = self.thisRunDirectory + '/GosplOutputFiles/h5/gospl.{}.p0.h5'
        gosplData = self.readGosplFile(gosplFilenamePattern.format(iteration))
        icoXYZ, icoCells = self.createIcosphere(subdivisions=self.subdivisions, radius=self.earth.earthRadius)
        outputMesh = self.createMeshFromNPZdata(icoXYZ, icoCells, gosplData['elev'], heightAmplification=self.earth.heightAmplificationFactor)
        
        #Store gospl data as mesh attributes
        outputMesh['elev'] = gosplData['elev'] 
        outputMesh['erodep'] = gosplData['erodep'] 
        outputMesh['fillAcc'] = gosplData['fillAcc']**0.25
        outputMesh['flowAcc'] = gosplData['flowAcc']**0.25
        outputMesh['rain'] = gosplData['rain'] 
        outputMesh['sedLoad'] = gosplData['sedLoad']**0.25
        return outputMesh
        
    #Given the data from the NPZ file, we create an pyvista mesh of earth
    @staticmethod
    def createMeshFromNPZdata(vertices, cells, heights, heightAmplification=30):
        faces = []
        exageratedRadius = vertices + heightAmplification * heights * vertices / np.max(vertices)
        for cell in cells:
            faces.append(3)
            faces.append(cell[0])
            faces.append(cell[1])
            faces.append(cell[2])
        earthMesh = pv.PolyData(exageratedRadius, faces)
        earthMesh['heights'] = heights
        return earthMesh
    
    #Animate the results produced by Gospl
    def animateGosplOutput(self,
                           scalarAttribute = 'elev',
                           gosplFilenamePattern = 'gospl.{}.p0.h5',
                           movieOutputFileName = 'GosplAnimation.mp4',
                           lookAtLonLat = np.array([60, 20]),
                           cameraZoom = 1.4,
                           framesPerIteration = 8,
                           runNumber = None):
        
        if (runNumber != None):
            self.thisRunDirectory = './{}/run{}'.format(self.mainOutputDirectory, runNumber)
        
        #Set up directories, get number of file for animation, and initialise the animation mesh
        outputDir = self.thisRunDirectory + '/GosplOutputFiles/h5/'
        outputPath = Path(outputDir)
        numOfFiles = len(list(outputPath.glob('gospl.*.p0.h5')))
        fileNamePattern = outputDir + gosplFilenamePattern
        earthMesh = self.createGosplDataMesh(0)
        
        #Set up plotter object and camera position for animation
        plotter = pv.Plotter()
        plotter.add_mesh(earthMesh, scalars=scalarAttribute, cmap='gist_earth')
        plotter.camera_position = 'yz'
        plotter.camera.zoom(cameraZoom)
        plotter.camera.azimuth = 180 + lookAtLonLat[0]
        plotter.camera.elevation = lookAtLonLat[1]
        plotter.show(auto_close=False, window_size=[800, 608])
        plotter.open_movie(movieOutputFileName)
        for i in range(framesPerIteration):
            plotter.write_frame()
        
        #Iterate through files and write animation frames
        for i in range(numOfFiles-1):
            newMesh = self.createGosplDataMesh(i+1)
            plotter.update_coordinates(newMesh.points, mesh=earthMesh)
            plotter.update_scalars(newMesh[scalarAttribute], render=False, mesh=earthMesh)
            for i in range(framesPerIteration):
                plotter.write_frame()
        plotter.close()
        return
    
    #The Gospl file contains simulation output data at particular iterations during the simulation
    @staticmethod
    def readGosplFile(fileDir):
        gosplDict = {}
        with h5py.File(fileDir, "r") as f:
            for key in f.keys():
                gosplDict[key] = np.array(f[key])
        return gosplDict