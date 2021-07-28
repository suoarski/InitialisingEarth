# Initialising Earth
A series of jupyter notebooks demonstrating python code for procedural earth surface dynamics with a particular focus on tectonic plates. This is still a work in progress and changes will be made regularly.


### Running the Notebooks with Docker

It is reccomended that these notebooks are run within a docker container. To do so, first download and install [Docker Desktop](https://www.docker.com/). Once installed, open CMD (for windows) and pull the earthinit image by running the following command:

```
docker pull suoarski/earthinit
```

Once the image has been pulled, clone a copy of the [InitialisingEarth Github](https://github.com/suoarski/InitialisingEarth) repository to some directory on your computer. Then, open Docker Desktop, under *Images* run the newly created docker image named *suoarski/earthinit* as follows:

![SomeText](Images/Docker1.PNG?raw=true "Title")

In the new pop up window, under *Option Settings* set the following attributes:

- **Container Name**: Earth Init (or any other name)
- **Local Host**: 1000 (or any other unused local port)
- **Host Path**: *C:\Path\To\InitialisingEarth\clone*
- **Container Path**: */live/share*

Then click run.

<div>
<img src="files/Images/Docker2.png" width="600">
</div>

This should create a new docker container. Clicking on *Open In Browser* should open a Jupyter Notebook environment in your local browser, which can be used to run the Initialising Earth notebook tutorials.

<div>
<img src="files/Images/Docker3.png" width="600">
</div>

If the container is properly set up, you should find the files of the cloned github repository by navigating to the *share* directory within the Jupyter Notebook environment. The notebook tutorials should then be found under the *NotebookTutorials* directory, where each tutorial has it's own sub directory.

