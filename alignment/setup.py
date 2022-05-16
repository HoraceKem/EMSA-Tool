import numpy as np  
from distutils.core import setup  
from distutils.extension import Extension  
from Cython.Distutils import build_ext  

ext_modules = [Extension("mesh_derivs_multibeam", ["mesh_derivs_multibeam.pyx"],include_dirs=[np.get_include()],extra_compile_args=['-fopenmp', '-O3', '--verbose'],extra_link_args=['-fopenmp'])]

setup(name="mesh_derivs_multibeam", cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
