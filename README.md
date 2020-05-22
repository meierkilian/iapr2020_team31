# Image Analysis and Pattern Recognition : Project, Team 31

## General information
This repository has be created to contain the final version of the project realised during the spring semester of 2020 in the Image Analysis and Pattern recognition course of [Prof. Jean-Philippe Thiran][jpt]. The repository containing the handout for this project can be found [here][handout]

[jpt]: https://people.epfl.ch/115534
[handout]: https://github.com/LTS5/iapr-2020

## Repository structure
* **data** : contains various data, in particular the original video to be processed
* **devSrc** : code used test and develop various feature of the software
* **env** : contains conda environement file to be used to install the virtuall environement
* **handin** : contains project presentation and software callgraph (used to have a overview of the software)
* **handout** : contains original project desription
* **src** : contains source code, in particular the main.py which is the program entry
* **README.md** : this file

## Installation
Assuming conda and git are already installed, tested on linux but should work on other OS if paths are adpated.

1. Download the repository
	```
	git clone https://github.com/meierkilian/iapr2020_team31.git
	```
2. Go in the repository
	```
	cd iapr2020_team31
	```
3. Install the conda environement
	```
	conda env create -f env/environment.yml
	```
4. Activate the conda environement 
	```
	conda activate iapr2020_team31
	```
5. Test installation, the help info should be displayed
	```
	python src/main.py -h
	```
6. Done


## Usage
```
main.py [-h] [-i INPUT] [-o OUTPUT] [-v]
```

optional arguments:
| Short     | Long            | Description                                                                                        |
|-----------|-----------------|----------------------------------------------------------------------------------------------------|
| -h        | --help          | Show this help message and exit                                                                    |
| -i INPUT  | --input INPUT   | Input video clip, should be .avi                                                                   |
| -o OUTPUT | --output OUTPUT | Output video clip (path and name), should be .avi                                                  |
| -v        | --verbose       | Makes processing verbose and displays intermediate figures (execution stops when a figure is open) |
## List of used packages
This list contains the aditional packages used in this project, however this list might not be exaustive so use the conda environement or read the env/environement.yml file.

* argparse
* av
* cv2
* keras
* matplotlib
* numpy
* os
* pickle
* PIL
* skimage
* sys
