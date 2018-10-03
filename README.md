Honours Project
Requirements
- All code written in Python 3.5
- Tested on Tensorflow 1.5.0 and EdwardLib 1.3.5
Usage
- All scripts can be edited to allow for:
	- Changing the file used from main
	- Changing the number of input and output variables from the flags 
Scrips
- 3Layer2OAuto.py: This script will train a 3 layer networks with 2 outputs on all permutations in the getParams method
- 3Layer4OAuto.py: This script will train a 3 layer networks with 4 outputs on all permutations in the getParams method
- 2Layer2OAuto.py: This script will train a 2 layer networks with 2 outputs on all permutations in the getParams method
- 2Layer4OAuto.py: This script will train a 2 layer networks with 4 outputs on all permutations in the getParams method
- 1Layer2OAuto.py: This script will train a 1 layer networks with 2 outputs on all permutations in the getParams method
- 1Layer4OAuto.py: This script will train a 1 layer networks with 4 outputs on all permutations in the getParams method

The files were seperated as Tensorflow does not remove graphs from memory when running multiple tests. This causes it to flood the memory with unused graphs.By seperating the files for each layer and experiment it allowed more tests to be run per layer and thus increasing the possibility of acheiving good results