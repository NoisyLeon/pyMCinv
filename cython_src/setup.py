from distutils.core import setup
from Cython.Build import cythonize
import numpy
import distutils.sysconfig
from distutils.extension import Extension
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

#setup(
#    ext_modules = cythonize("data.pyx", language='c++'),
#    include_dirs=[numpy.get_include()]
#)


#setup(
#    name = 'pymcinv',
#    ext_modules = cythonize(["*.pyx"], language='c++'), 
#    include_dirs=[numpy.get_include()],
#    extra_compile_args=["-std=c++11"]
#)

ext_modules = [
     Extension(
         'data',
         ["data.pyx"],
         extra_compile_args=['-fopenmp'],
         extra_link_args=['-fopenmp']
     ),
    Extension(
        'modparam',
        ["modparam.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    ),
     Extension(
         'vmodel',
         ["vmodel.pyx"],
         extra_compile_args=['-fopenmp'],
         extra_link_args=['-fopenmp']
     ),
     
     Extension(# module name:
     'invsolver',
     # source file:
     ['invsolver.pyx'],
     # other compile args for gcc
     extra_compile_args=['-fPIC', '-O3', '-lgfortran', '-fopenmp'],
     # other files to link to
     extra_link_args=['./fast_surf_src/calcul.o', './fast_surf_src/fast_surf.o', './fast_surf_src/flat1.o', './fast_surf_src/init.o',
                     './fast_surf_src/mchdepsun.o', './fast_surf_src/surfa.o',
                     './rftheo_src/four1.o', './rftheo_src/qlayer.o', './rftheo_src/theo.o', 
                     '-fPIC', '-O3', '-lgfortran', '-fopenmp']
     )
    
]

setup(
    name = 'pymcinv',
    ext_modules = cythonize(ext_modules), 
    include_dirs=[numpy.get_include()]
)


