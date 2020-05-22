=================================================================================================
============                 Keras Environment SETUP                                 ============   
=================================================================================================

Here is the complete setup to follow in order to have a Keras environment with anaconda prompt.
1) You first need to install either anaconda (https://docs.anaconda.com/anaconda/install/) or 
miniconda (https://docs.conda.io/en/latest/miniconda.html). Once it's done, you'll have access
to the anaconda prompt. 
2) Open the anaconda prompt and then write the following commands : 


conda create --name PythonGPU                   // or for CPU : conda create --name PythonCPU

conda activate PythonGPU                        // or for CPU : conda activate  PythonCPU

conda install python=3.6

//Verifiez que vous avez bien la bonne version : 
python --version

conda install -c anaconda keras-gpu             // or for CPU : conda install -c anaconda keras

conda install -c conda-forge scikit-image       // maybe this one is not necessary
conda install -c conda-forge opencv
conda install --file install2.txt

conda install jupyter notebook

//just type the following command to launch the jupyter notebook :
jupyter notebook

