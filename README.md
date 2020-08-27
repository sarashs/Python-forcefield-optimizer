# PyField version 0.0

 This is a forcefield optimization tool. This tool allows you to optimize your classical forcefield based on Quantum Chemistry calculations. At this point, the following forcefields are supported: 
- REAX-FF

Also the following features are implemented:

- Supported OS
    - Linux (tested on Ubuntu 20.04)
    - Also works with windows linux subsystem (tested on Ubuntu 20.04)
- Modes of execution
    - Serial execution: It is good for small molecular systems. We will soon release the parallel version.
- Types of optimization
    - Energy
    - Charge
- Optimization methods
    - ##### Simulated annealing
    - Genetic Algorithm

If you want to use this tool please read the following carefully. Also, if you publish anything using our tool, please properly `cite` our paper as explained in [Citation](#team).

## Table of Contents
- [Architecture](#architecture)
- [Installation](#installation)
- [Examples](#examples)
- [Features](#features)
- [Citation](#team)
- [License](#license)
## Architecture
This tool has three main classes:
1. The **forcefield** Class: This class is a parrent of every forcefield class (currently we only have the ReaxFF forcefield)
2. The **SA** or simulated annealing Class: This is the parrent simulated annealing class which has daughters for various forcefields based on their needs.
3. The **GA** or Geneteics Algorithm Class: This is the parrent genetic algorithms class.
The tool also comes with a **LAMMPS_Utils** package which has various functions for for preparing and running LAMMMPS simulations. 
## Installation
- This tool was built in python 3.6. We recommend that you set up an environment with python 3.6.
- This tool relies on LAMMPS in order to perform structural minimizations as well as charge calculations. Therefore, before using this tool **you have to build LAMMPS as a shared library** for python on your linux machine. Please refer to <a href="https://lammps.sandia.gov/doc/Python_overview.html" target="_blank">`Lammps documentation`</a> for more instructions.
### Clone and requirement installation
After the above steps you simply clone this repository: 
```shell 
$ git clone `repo name`
$ pip install requirements.txt
```
### Testing
In order to test the program, simply navigate to the Pyfield folder and execute the following command:
```shell 
$ python Pyfield.py tests/Pyfield_input.pyfield 
```
After the execution of this command, a **pyfield_logs.txt** file is created. By navigating through that file, you can identify if there is any problem otherwise, it should print "Success" in front of all of the tests. 
## Examples
There are two examples in the tests folder. One is for the Cl-Cl bond and the other is for the Hydrogen bonds (both with reax forcefields). The following commands will run them for you:
```shell 
$ python Pyfield.py tests/Cl2.pyfield 
```
and
```shell 
$ python Pyfield.py tests/Hbonds.pyfield 
```
These two examples, along with the input, param, structure and training files should be a good starting point.
## Features
To be completed later.
## `Citation`
Please cite the following article where this tool was first introduced: **Our article to come!**
## License
