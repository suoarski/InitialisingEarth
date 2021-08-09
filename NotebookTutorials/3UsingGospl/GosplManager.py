
#Class for seting up files for and running Gospl
class GosplManager:
    def __init__(self, earth,
                subdivisions=6,
                 mainOutputDirectory = 'GosplRuns'
                ):
        
        #Store attributes passed by class initiation and directory specifications
        self.earth = earth
        self.subdivisions = subdivisions
        self.mainOutputDirectory = mainOutputDirectory
        self.forwardOutputFolder = '/GosplOutputFiles'
        self.backwardOutputFolder = '/BackwardsOutput'
        
        #Icosphere for the creation of npz files
        self.icosphereXYZ, self.icoCells = self.createIcosphere(subdivisions=subdivisions, radius=earth.earthRadius)
        
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
        self.gosplStepsAtEachIteration = 5
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
        self.uniform = 1.
        
        #ForcePaleo attributes
        self.steps = [int(earth.startTime), int(earth.endTime)]#, int(earth.deltaTime)]
        
        #Output Attributes
        self.dirFormat = 'GosplOutput{}'
        self.makedir = False
        
    #============================== YML File Creation ======================================================
    #We generate a string containing all the content of the backwards and forwards YML file
    #For each section of the YML file, we have a seperate function creating a string for it,
    #We then combine all these sections into a single string and create the YML file
    def getNameString(self):
        return "\nname: {}\n\n".format(self.name)
    
    def getDomainString(self, backwards=False):
        domainFormat = "domain:\n  npdata: '{}'\n  flowdir: {}\n  fast: {}\n  backward: {}\n  interp: {}\n\n"
        npdataFormat = self.npdataFormat
        time = earth.startTime
        if backwards:
            npdataFormat = self.npdataBackFormat
            time = earth.endTime
        npdata = npdataFormat.format(self.npzFilesDirectory, time)
        domainString = domainFormat.format(
            npdata,
            self.flowdir,
            backwards,
            backwards,
            self.interp)
        return domainString
    
    def getTimeString(self):
        timeFormat = "time:\n  start: -{}\n  end: {}\n  tout: {}\n  dt: {}\n  tec: {}\n\n"
        self.dt = earth.deltaTime * 1000000 / self.gosplStepsAtEachIteration
        timeString = timeFormat.format(
            float(self.start),
            float(self.end),
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
                1000000 * (time - earth.deltaTime),
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
                1000000 * (time - earth.deltaTime),
                self.npzFilesDirectory,
                self.subdivisions,
                np.max(times) - time + earth.deltaTime)
            tectonicString += tectonicPart
        return tectonicString + '\n'
    
    def getClimateString(self):
        climateFormat = "climate:\n  - start: -{}.\n    uniform: {}\n\n"
        climateString = climateFormat.format(
            1000000 * self.earth.startTime,
            self.uniform)
        return climateString
    
    def getForcePaleoString(self):
        forcePaleoFormat = "forcepaleo:\n  dir: '{}'\n  steps: {}\n\n"
        forcePaleoString = forcePaleoFormat.format(
            self.thisRunDirectory + '/' + self.backwardOutputFolder,
            self.steps)
        return forcePaleoString
    
    def getOutputString(self, backwards=False):
        outputFormat = "output:\n  dir: '{}'\n  makedir: {}\n\n"
        outputFolder = self.forwardOutputFolder
        if backwards:
            outputFolder = self.backwardOutputFolder
        outputString = outputFormat.format(
            self.thisRunDirectory + outputFolder,
            self.makedir)
        return outputString
    
    #Create a string containing all the content of the YML file
    def getForwardYMLstring(self):
        name = self.getNameString()
        domain = self.getDomainString()
        time = self.getTimeString()
        spl = self.getSPLstring()
        diffusion = self.getDiffusionString()
        sea = self.getSeaString()
        tectonic = self.getForwardTectonicString()
        climate = self.getClimateString()
        forcePaleo = self.getForcePaleoString()
        output = self.getOutputString()
        return name + domain + time + spl + diffusion + sea + tectonic + climate + forcePaleo + output
    
    #Create a string containing all the content of the YML file
    def getBackwardYMLstring(self):
        name = self.getNameString()
        domain = self.getDomainString(backwards=True)
        time = self.getTimeString()
        spl = self.getSPLstring()
        diffusion = self.getDiffusionString()
        sea = self.getSeaString()
        tectonic = self.getBackwardsTectonicString()
        climate = self.getClimateString()
        output = self.getOutputString(backwards=True)
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
        
        #Create the backwards YML file
        ymlBackwardContent = self.getBackwardYMLstring()
        self.ymlBackwardDirectory = thisRunDirectory + '/Backward.yml'
        ymlBackward = open(self.ymlBackwardDirectory, 'w')
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
    
    #Create the domains npdata elevations file
    def createDomainNPdataFile(self):
        
        #Create data file for forward model
        heights = self.getIcoHeights(iteration=0)
        neighbours = self.getNeighbourIds()
        fileName = self.npdataFormat.format(self.npzFilesDirectory, earth.startTime)
        np.savez_compressed(fileName, v=self.icosphereXYZ, c=self.icoCells, n=neighbours.astype(int), z=heights)
        
        #Create data file for backward model
        heights = self.getIcoHeights(iteration=-1)
        fileName = self.npdataBackFormat.format(self.npzFilesDirectory, earth.endTime)
        np.savez_compressed(fileName, v=self.icosphereXYZ, c=self.icoCells, n=neighbours.astype(int), z=heights)
    
    #============================== Create Tectonic Displacements Files =============================================
    #We interpolate the force field onto an Icosphere suitable for Gospl
    def interpolateForces(earthLonLat, icosphereXYZ, forceXYZ):
        radLonLat = EarthAssist.cartesianToPolarCoords(icosphereXYZ)
        icoLonLat = np.stack((radLonLat[1], radLonLat[2]), axis=1)
        icoForce = interpolate.griddata(earthLonLat, forceXYZ, icoLonLat, method='cubic')
        return icoForce
    
    #Create tectonic force displacement files
    def createTectonicDispNPZfiles(self):
        times = self.earth.simulationTimes[:-1]
        for i, time in enumerate(times):
            tectonicDisp = self.earth.tectonicDisplacementHistory[i]
            icoForce = interpolateForces(self.earth.lonLat, self.icosphereXYZ, tectonicDisp)
            
            #Forward displacement files
            fileName = '{}/ForceSubdivisions{}Time{}Mya'.format(self.npzFilesDirectory, self.subdivisions, time)
            np.savez_compressed(fileName, xyz=icoForce)
            
            #Backward displacement files
            fileName = '{}/BackwardsForceSubdivisions{}Time{}Mya'.format(self.npzFilesDirectory, self.subdivisions, time)
            np.savez_compressed(fileName, xyz=-icoForce)
    
    #Run the gospl simulation
    def runGosplSimulation(self):
        
        #Run backward model
        mod = sim(self.ymlBackwardDirectory, False, False)
        mod.runProcesses()
        mod.destroy()
        
        #Run forward model
        mod = sim(self.ymlForwardDirectory, False, False)
        mod.runProcesses()
        mod.destroy()
    
    #After creating a GosplManager object, this function will do everything to run a simulation
    #Alternatively you can call these functions individually
    def createAllFilesAndRunSimulation(self):
        self.makeDirectories()
        self.createDomainNPdataFile()
        self.createTectonicDispNPZfiles()
        self.runGosplSimulation()