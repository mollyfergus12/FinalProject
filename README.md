# FinalProject
Repository for final assessment for module egm722. A primary analysis of Northern Ireland health problem and deprivation data. 

**1. How to Install Code**

To run this code, the installation of Github, Git Desktop, Anaconda and an Integrated Development (IDE), such as Pycharm, is advised. Once these are installed, this repository can be cloned to your computer. 

The FinalProject repository will be required to be forked to your own repository. Log into or create a github account and go to https://github.com/mollyfergus12/FinalProject. Click the Fork button on the upper right hand corner of the window. This will copy the entire repository to your account.

Following this, open Git and ensure you are signed into your Github account. Select FIle > Clone repository to copy your fork of the FinalProject repository. Select the URL tab, paste the URL for your fork of the FinalProject repository. Save this to a folder on your computer and select Clone. When asked how you are planning to use the fork, click ‘For my own purposes’ and continue. The repository, and the datafiles, py. File, environment.yml file should now be downloaded on your computer. For purposes of clarity it is recommended to place all datafiles into one subfolder named ‘Datafiles’. See section 2.4 for more information.

Once you have cloned the repository, a conda environment can then be created on Anaconda that will provide the modules necessary to run the code. To create the conda environment, use the environment.yml file provided in the GitHub repository. On Anaconda, the environment can be created by opening Anaconda, selecting import from the Environments Panel on the bottom left-hand side of the screen, and selecting the environment.yml file that was downloaded from the FinalProject repository. 

To run the code, return to PyCharm and select the FinalProject.py file to open. Ensure that it is set to the correct configuration (the location of your repository on your computer) and the interpreter is set to the repository’s environment in order to run the file successfully. 

**2. Sourcing and Downloading datasets**

The datafiles used within this code are uploaded to the FinalProject repository for ease of use. The health problem and deprivation shapefiles exceed the size permitted to be directly uploaded to github via user interface, so a file is available with the links to a Google Drive containing these shapefiles for download. Having copied the repository to your computer on Git, the remaining datasets should be located in the folder you created for the repository. The datafiles were divided into two folders in order to upload to the repository, so for purposes of clarity it is recommended to place all datafiles, including the shapefiles available to download from the Google Drive link, into a subfolder named ‘Datafiles’ in your repository folder. 

The health problem data was sourced from: http://infuse.mimas.ac.uk/help/definitions/2011geographies/index.html
Under Statistics>Census 2011>Filters:Subset: Health. Under Geography> SA> “Long-Term Health Problem or Disabilitity_QS303NI (statistical geographies). The file was then cleaned and columns renamed for clarity, and then joined to a Small Area ESRI shapefile sourced from:https://www.nisra.gov.uk/support/geography, using ArcGIS Pro. The Multiple Deprivation Measure data is originally sourced from: https://www.nisra.gov.uk/, under Statistics>Deprivation>Northern Ireland Multiple Deprivation Measure 2017 (NIMDM2017)>SA level results. The data was then cleaned and modified and joined to a Small Area ESRI shapefile using ArcGis Pro. As the Health_Issue and MDM shapefiles were too large to be directly uploaded via the github user interface, Git LFS was used to upload them to the repository. 

The counties and water shapefiles were also sourced from OSNI Open Data, under “Largescale Boundaries County Boundaries” and “Northern Ireland Lakes Water Bodies”, available from:https://www.opendatani.gov.uk/dataset?q=water&res_format=SHP&sort=score+desc%2C+metadata_modified+desc.



