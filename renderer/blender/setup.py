import numpy as np
import subprocess
import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from distutils.core import setup
# from distutils.extension import Extension
from Cython.Distutils import build_ext  

flags = subprocess.check_output(['pkg-config', '--cflags-only-I', 'opencv4'])
include_dirs_list = [str(flag[2:].decode('utf-8')) for flag in flags.split()]
include_dirs_list.append('.')
include_dirs_list.append(np.get_include())
flags = subprocess.check_output(['pkg-config', '--libs-only-L', 'opencv4'])
library_dirs_list = flags
flags = subprocess.check_output(['pkg-config', '--libs', 'opencv4'])
libraries_list = []
for flag in flags.split():
    libraries_list.append(str(flag.decode('utf-8')))

ext_modules = [Extension("images_composer",
                         ["images_composer.pyx",
                          "ImagesComposer.cpp",
                          os.path.join("detail", "seam_finders.cpp"),
                          os.path.join("detail", "exposure_compensate.cpp"),
                          os.path.join("detail", "blenders.cpp")],
                         language="c++",
                         include_dirs=include_dirs_list,
                         extra_compile_args=['-O3', '--verbose'],
                         extra_objects=libraries_list)
               ]

setup(name="images_composer", cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
