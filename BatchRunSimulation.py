
import InitialisingEarth as earthInit
import numpy as np
from scipy.interpolate import *


#Create a dictionary of properties that will be passed to the main code
props = {}

#We define a few template curves here
#When manipulating these curves, makes sure to set 'showTemplate' variables to true 

#Template curve for the distance transfer for subduction uplift
upliftTemplatePoints = np.array([
    [-100, 0.0],
    [-1.0, 0.0],
    [-0.101, 0.0],
    [-0.1, 0.0],
    [0, 0.4],
    [0.19, 1.0],
    [0.21, 1.0],
    [0.5, 0.5],
    [0.99, 0.0],
    [1.0, 0.0],
    [100.0, 0.0]
    ])
props['upliftTemplate'] = interp1d(upliftTemplatePoints[:, 0], upliftTemplatePoints[:, 1], kind='quadratic')
showUpliftTemplate = False

#Template curve for continental shelves
continentalShelfPoints = np.array([
    [0, 0],
    [0.01, -0.04],
    [0.25, -0.1],
    [0.49, -0.196],
    [0.5, -0.2],
    [0.56, -1.0],
    [0.7, -4.5],
    [0.75, -4.66666666],
    [1.0, -5.5],
    [1.01, -5.5],
    [4.0, -5.5],
    [9.9, -5.5],
    [10, -5.5],
    [1000, -5.5]
    ])
continentalShelfPoints[:, 1] /= 5.5
props['continentalShelfTemplate'] = interp1d(continentalShelfPoints[:, 0], continentalShelfPoints[:, 1], kind='quadratic')
showContShelfTemplate = False

#============================================== Simulation Properties ==================================================
##Directory for data files
props['platePolygonsDirectory'] = './Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies_PMAG.gpmlz'
props['platePolygonsDirectory400MYA'] = './Matthews_etal_GPC_2016_Paleozoic_PlateTopologies_PMAG.gpmlz'
props['rotationsDirectory'] = './Matthews_etal_GPC_2016_410-0Ma_GK07_PMAG.rot'
props['coastLinesDirectory'] = './Matthews_etal_GPC_2016_Coastlines.gpmlz'

#Set the time range of simulation and time steps
props['timeFrom'], props['timeTo'], props['deltaTime'] = 100, 0, 5

#General earth properties
props['resolution'] = 100 #Resolution of sphere
props['earthRadius'] = 6371 #Radius of the earth in kilometres
props['heightAmplificationFactor'] = 60 #For visualization, we amplify the height to make topography more visible

#Properties for subduction
props['maxUpliftRange'] = 700 #Maximum distance that our subduction algorithm has an effect on
props['maxMountainHeight'] = 8 #Max mountain height (Only used in earth initialization)
props['baseSubductionUplift'] = 0.3 #Defines how fast mountains grow due to subduction
props['numberToAverageOver'] = 40 #The speed transfer takes the average collision speed of several nearby subduction regions
props['pointToPassThrough'] = (0, 0.05) #Our height transfer is defined by a sigmoid that passes through this point
props['centre'] = 3.0 #Our height transfer is a sigmoid with this as it's centre
showHeightTransfer = False

#Used for defining depth of oceanic ridges
props['oceanDepth'] = 5.5
props['ridgeDepth'] = 2.5 
props['ageProportion'] = 0.35
props['oceanShelfLength'] = 400 #The length of oceanic shelves
props['isOceanThreshold'] = -1.0 #For the purpose of our algorithm, we identify oceans to be lower than 0, it gives better results

#When continents diverge, we lower the land around the diverging boundary until it becomes an ocean
props['divergingContinentLoweringRange'] = 80
props['loweringRate'] = 0.4

#Melting mountains refers to mountains collapsing under their own weight over time
props['meltingRate'] = 0.005
props['weightInfluenceRange'] = 1000
props['meltingSigmoidCentre'] = 500
props['meltingSigmoidSteepness'] = 1/140
showMeltingProfile = False

#Some control parameters for how we want the simulation to run
props['autoRunSimulation'] = True #If false, the user must press 'q' to advance a step in the simulation
props['saveAnimation'] = True #Do we want to save the simulation as an mp4 file?
props['showMainPlot'] = False #Keep this as false to avoid having to press 'q' on each run of the simulation
props['writeDataToFile'] = True #Do we want to save data as a csv file?

#If we choose to view any of the profile curves, we run this function instead of the main code
#Note that we don't run the main code when displaying template curves
earthInit.showAlternativePlots(props, showUpliftTemplate, showContShelfTemplate, showHeightTransfer, showMeltingProfile)

if __name__ == '__main__':
    earthInit.runMainTectonicSimulation(props)
    
    props['resolution'] = 400
    #earthInit.runMainTectonicSimulation(props)
    
    props['timeFrom'] = 200
    props['timeTo'] = 100
    #earthInit.runMainTectonicSimulation(props)
    
    props['deltaTime'] = 1
    props['timeTo'] = 0
    #earthInit.runMainTectonicSimulation(props)
    