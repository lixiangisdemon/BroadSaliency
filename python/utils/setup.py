from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
import sys
import os
import glob

prefix = '/Users/lixiang/Desktop/BroadSaliency'
Trdprefix = prefix + '/3rdparty'
lib_folder = os.path.join(prefix, 'libs')
if (sys.platform == 'darwin'):
    sys_include = '/Library/Developer/CommandLineTools/usr/include/c++/v1'
    extra_link_args=["-stdlib=libc++", "-mmacosx-version-min=10.9"]
else:
    sys_include = '/usr/local/include'
    extra_link_args=None
    
include_dirs = [
                sys_include,
                numpy.get_include(),
                os.path.join(Trdprefix, 'eigen3'),
                os.path.join(Trdprefix, 'densecrf/include'),
                os.path.join(prefix, 'featureExtra/include'),
                os.path.join(prefix, 'supercrf/include')
                ]

extension1 = Extension("opencv_mat",
                sources=['opencv_mat.pyx'],
                language="c++",
                include_dirs=include_dirs,
                library_dirs=[lib_folder],
                libraries=['opencv_core', 'opencv_imgproc', 'opencv_ml'],
                extra_link_args=extra_link_args,
                extra_compile_args = ["-std=c++14"]
                )

extension2 = Extension("extrafeature",
                sources=["extrafeature.pyx"],
                language="c++",
                include_dirs=include_dirs,
                library_dirs=[lib_folder],
                libraries=['opencv_core', 'opencv_imgproc', 'opencv_imgcodecs', 'opencv_highgui', 'opencv_ml', 'featureExtra'],
                extra_link_args=extra_link_args,
                extra_compile_args = ["-std=c++14"]
                )
extension3 = Extension("supercrf",
                sources=['supercrf.pyx'],
                language="c++",
                include_dirs=include_dirs,
                library_dirs=[lib_folder],
                libraries=['opencv_core', 'opencv_imgproc', 'opencv_imgcodecs', 'opencv_highgui', 'opencv_ml', 'densecrf', 'featureExtra', 'supercrf'],
                extra_link_args=extra_link_args,
                extra_compile_args = ["-std=c++14"]
                )

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[extension1, extension2, extension3]
)

if (sys.platform == 'darwin'):
    os.system('install_name_tool -change @rpath/liboptimization.dylib {}/libs/liboptimization.dylib {}/libs/libsupercrf.dylib'.format(prefix, prefix))
    os.system('install_name_tool -change @rpath/libfeatureExtra.dylib {}/libs/libfeatureExtra.dylib extrafeature.cpython-37m-darwin.so'.format(prefix))
    os.system('install_name_tool -change @rpath/libfeatureExtra.dylib {}/libs/libfeatureExtra.dylib supercrf.cpython-37m-darwin.so'.format(prefix))
    os.system('install_name_tool -change @rpath/libdensecrf.dylib {}/libs/libdensecrf.dylib supercrf.cpython-37m-darwin.so'.format(prefix))
    os.system('install_name_tool -change @rpath/libsupercrf.dylib {}/libs/libsupercrf.dylib supercrf.cpython-37m-darwin.so'.format(prefix))
