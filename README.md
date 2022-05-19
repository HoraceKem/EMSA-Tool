# EMSA-Tool
## I Introduction
A tool for large-scale EM images stitching and alignment. This repo is a secondary development based 
on [rh-aligner](https://github.com/Rhoana/rh_aligner) and [rh-renderer](https://github.com/Rhoana/rh_renderer)
(Copyright@Adi Peleg, VCG Group, Harvard University). Compared to the Adi's version, EMSA-Tool has 
the following modifications: 
1. **Up-to-date.** All the Python2 codes are deprecated and have been rewritten in the Python3 style. 
The important dependence OpenCV also upgrades from 2.x to the latest 4.5.
2. **Unified style names.** The S&A processing contains many I/O operations and proper nouns.
Here we establish a dictionary covering most of the word in variables and functions. The naming style follows 
the Google Style Guide. e.g. lowercase_with_underline for variables and UpperCamelCase for classes.
3. **Multiple EM types.** It supports both singlebeam and multibeam EM images' data structure.
4. **Auto-generated documents.** Using Sphinx style to generate the documents for better reading and development.
5. **Logging management.** We replace the print function by a log controller to output and save hierarchical logs.
6. **Unified arguments.** All the adjustable arguments during the processing are collected in the 'arguments' folder.
7. **Acceleration.** Different from the parallel computation using cluster strategy in rh-aligner, EMSA-Tool focuses on
the PC environment and provides multiple acceleration methods: numba jit, numba cuda jit, multiprocessing, cython.


## II Environment
Because of the potential dependencies' problem, we strongly recommend you preparing a brand new virtual 
environment(here we use Anaconda) as follows:  
*Tip: THE_UPPERCASE_WITH_UNDERLINE should be changed by yourself.*
1. Create a Python virtual environment and activate it:
```shell
conda create -n EMSA python=3.9
conda activate EMSA
```
2. (Optional) Using ```nvidia-smi``` to confirm your CUDA version and prepare the CUDA in the python environment:
```shell
conda install cudatoolkit=YOUR_CUDA_VERSION
```
3. Install the pypi packages:
```shell
cd YOUR_PATH_TO_EMSA_TOOL
pip install -r requirements.txt
```
4. Install OpenCV:  
Because of the non-free and CUDA version algorithms, we need to build OpenCV-Python from the source instead of
installing from conda or pip. First download the [opencv](https://github.com/opencv/opencv/releases) 
and [opencv-contrib](https://github.com/opencv/opencv_contrib/tags) source codes and unzip them. 
Version 4.5.5 is recommended. Keep your EMSA environment activated, and run the following command 
in the folder containing opencv and opencv_contrib:
```shell
mkdir build
cd build
```
```shell
cmake -D CMAKE_BUILD_TYPE=RELEASE 
-D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.5.5/modules 
-D OPENCV_GENERATE_PKGCONFIG=ON 
-D WITH_CUDA=ON 
-D OPENCV_ENABLE_NONFREE=ON 
-D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") 
-D PYTHON3_EXECUTABLE=$(which python) 
-D PYTHON3_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") 
-D PYTHON3_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") 
../opencv-4.5.5
```
```shell
make -j8
make install
```
```shell
cd YOUR_PATH_TO_ANACONDA/envs/EMSA/lib/python3.9/site-packages
ln -s ./cv2/python3.9/YOUR_CV2_SO_FILE_NAME cv2.so
```
Here, YOUR_CV2_SO_FILE_NAME should be similar to 'cv2.cpython-39-x86_64-linux-gnu.so' in Linux.  

5. Compile cython files and generate .so files which can be imported by Python:
```shell
cd YOUR_PATH_TO_EMSA_TOOL/alignment
python setup.py build_ext --inplace
cd ../renderer/blender
export PKG_CONFIG_PATH=YOUR_PATH_TO_ANACONDA/envs/EMSA/lib/pkgconfig:$PKG_CONFIG_PATH
python setup.py build_ext --inplace
```
6. Install tinyr:
Please refer to the [tinyr](https://github.com/HoraceKem/tinyr) repo.
