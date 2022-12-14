# Initialising Earth
A series of jupyter notebooks demonstrating python code for simulating plate tectonics over the course of millions of years. Note that this was some of my earlier works in a research project that then later evolved to my contribution in a publication in the jurnal "Nature Reviews Earth & Environment" titled "[A glimpse into a possible geomorphic future of Tibet](https://www.nature.com/articles/s43017-022-00355-z)"

# Video Presentation
[See this youtube video for a presentation on this project](https://youtu.be/F_suI-KvDiI)

# Notebooks
- [Moving Tectonic Plates](https://github.com/suoarski/InitialisingEarth/blob/main/NotebookTutorials/1MovingTectonicPlates/1MovingPlates.ipynb)
- [Subduction Uplift](https://github.com/suoarski/InitialisingEarth/blob/main/NotebookTutorials/2SubductionUplift/2TectonicForces.ipynb)
- [Diverging Plates](https://github.com/suoarski/InitialisingEarth/blob/main/NotebookTutorials/3Divergence/3Divergence.ipynb)
- [Using GOSPL](https://github.com/suoarski/InitialisingEarth/blob/main/NotebookTutorials/4UsingGospl/4UsingGospl.ipynb)


### Running the Notebooks with Docker

It is reccomended that these notebooks are run within a docker container. To do so, first download and install [Docker Desktop](https://www.docker.com/). Once installed, open CMD (for windows) and pull the earthinit image by running the following command:

```
docker pull suoarski/earthinit
```

Once the image has been pulled, clone a copy of the [InitialisingEarth Github](https://github.com/suoarski/InitialisingEarth) repository to some directory on your computer. Then, open Docker Desktop, under *Images* run the newly created docker image named *suoarski/earthinit* as follows:

![DockImage](Images/Docker1.PNG?raw=true "Title")

In the new pop up window, under *Option Settings* set the following attributes:

- **Container Name**: Earth Init (or any other name)
- **Local Host**: 1000 (or any other unused local port)
- **Host Path**: *C:\Path\To\InitialisingEarth\clone*
- **Container Path**: */live/share*

Then click run.

![DockerImage](Images/Docker2.png?raw=true "Title")

This should create a new docker container. Clicking on *Open In Browser* should open a Jupyter Notebook environment in your local browser, which can be used to run the Initialising Earth notebook tutorials.

![DockerImg](Images/Docker3.png?raw=true "Title")

If the container is properly set up, you should find the files of the cloned github repository by navigating to the *share* directory within the Jupyter Notebook environment. The notebook tutorials should then be found under the *NotebookTutorials* directory, where each tutorial has it's own sub directory.

