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
the PC environment and provides multiprocessing.


## II Environment
EMSA-Tool is now tested on Linux platform (Ubuntu 18.04), so currently we cannot ensure the 
compatibility of Windows nor macOS.  
Because of the potential dependencies' problem, we strongly recommend you preparing a brand new virtual 
environment(here we use Anaconda) as follows:  
*Tip: THE_UPPERCASE_WITH_UNDERLINE should be changed by yourself.*
1. Create a Python virtual environment and activate it:
```shell
conda create -n EMSA python=3.9
conda activate EMSA
```
2. Using ```nvidia-smi``` to confirm your CUDA version and prepare the CUDA in the python environment:
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

## III How to use
### Prepare your data
EMSA-Tool supports both singlebeam and multibeam EM data. For large-scale singlebeam dataset containing 
multiple imaging folders, your should first ensure that each section has one distinctive global index.
We provide one singlebeam dataset and one multibeam dataset for tests. You can download and unzip them into
the data folder.  
[singlebeam_test_dataset]: https://drive.google.com/file/d/1bdAiYg0YN21MZDH2VD0YSIunZ05HkZgt/view?usp=sharing
[multibeam_test_dataset]: TODO
### Set the running arguments
All the arguments are stored in the 'arguments' folder. The arguments will be copied to the workspace 
before the script running in case you forget the running arguments after a long time.
### Run the script
Just run it without argparse:
```commandline
python run.py
```

## IV Arguments 
In the 'arguments' folder, you can see three json files. Among them, features_args and align_args are 
algorithm details and here we just explain the overall_args.

|             arguments              |                                                                            explanation                                                                             |
|:----------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|            running_mode            |                                     ["release", "debug"] In 'debug' mode, multiprocess will be disabled for better debugging.                                      |
|              EM_type               |                          ["singlebeam", "multibeam"] The EM type you declared here should match the data, or an exception will be raised.                          |
|             workspace              |                                        [str] The workspace will store all the files which generated by the running script.                                         |
|           sample_folder            |            [str] Sample folder is the path to the target dataset. <br/>You have to ensure that the next level of sample folder is the section folders.             |
|            sample_name             |                                                                   [str] The name of the sample.                                                                    |
|           skipped_layers           | [str] The layers you want to skip. e.g. "5-7" means that layer-5, 6, 7 will not be included in the S&A processing. <br/>Multiple ranges should be connected by ',' |
| multiprocess(containing four args) |                [int] The number of threads you want to use in different steps.<br/> You should set them according to your machines' CPU and memory.                |
|           features_type            |                [str] The method of extracting the keypoint features. <br/>Please refer to common/keypoint_features_extraction.py for more details.                 |
|           matching_type            |                   [str]The method of matching the keypoint features. <br/>Please refer to common/keypoint_features_matching.py for more details.                   |
|         max_layer_distance         |      [int] An argument for elastic alignment algorithm. <br/>Please refer to Saalfeld's [paper](https://www.nature.com/articles/nmeth.2072) for more details.      |
|                mip                 |      [int] In short of mipmap level. 0 means you will render the result in the original resolution. <br/>1 means you will down-sample it by 2. scale=1/2^mip       |
|     from_x, from_y, to_x, to_y     |                 [int] The rendering range. [0, 0, -1, -1] means it will render the whole bounding box <br/>, which is decided by the S&A results.                  |
|             tile_size              |                                       [int] The size of the rendered images. e.g. 4096 means each image's size is 4096*4096                                        |
|             file_type              |                                                            [str] The extension of the rendered images.                                                             |
|            invert_image            |                                                   [0, 1] If you want to invert the raw EM images when rendering.                                                   |
|             blend_type             |                                       [0, 1, 2, 3] Referring to no_blending/averaging/linear/multi_band_seam, respectively.                                        |

## Contact
E-mail: horacekem@163.com  
Author: Hongyu Ge  
Biomed ssSEM Lab, SIBET  
Keling Road 88, Suzhou, Jiangsu