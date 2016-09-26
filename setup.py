#! /usr/bin/python
"""Setuptools-based setup script for ForcePy.

A working installation of NumPy <http://numpy.scipy.org> is required.

For a basic installation just type the command::

  python setup.py install

This script was adapted from MDAnalysis setup.py by Naveen Michaud-Agrawal.
"""
from setuptools import setup, Extension
from distutils.ccompiler import new_compiler
#
#------------------------------------------------------------

import sys, os
import glob

# Make sure I have the right Python version.
if sys.version_info[:2] < (2, 5):
    print "ForcePy requires Python 2.5 or better. Python %d.%d detected" % \
        sys.version_info[:2]
    print "Please upgrade your version of Python."
    sys.exit(-1)

try:
    # Obtain the numpy include directory.  This logic works across numpy versions.
    import numpy
except ImportError:
    print "*** package 'numpy' not found ***"
    print "MDAnalysis requires a version of NumPy (>=1.0.3), even for setup."
    print "Please get it from http://numpy.scipy.org/ or install it through your package manager."
    sys.exit(-1)

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

include_dirs = [numpy_include]

# Handle cython modules
try:
    from Cython.Distutils import build_ext
    use_cython = True
    cmdclass = {'build_ext': build_ext}
except ImportError:
    use_cython = False
    cmdclass = {}

if use_cython:
    # cython has to be >=0.16 to support cython.parallel
    import Cython
    required_version = "0.16"
    cython_version = ''.join(c for c in Cython.__version__ if c.isdigit())
    if not int(cython_version) >= int(required_version.replace(".", "")):
        raise ImportError("Cython version %s (found %s) is required because it offers a handy parallelisation module" % (required_version, Cython.__version__))
    del Cython

if __name__ == '__main__':
    RELEASE = "0.1"
    with open("README.md") as summary:
        LONG_DESCRIPTION = summary.read()
    CLASSIFIERS = ['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: POSIX',
                   'Operating System :: MacOS :: MacOS X',
                   'Programming Language :: Python',
                   'Programming Language :: C',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Topic :: Scientific/Engineering :: Chemistry',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   ]

    if 'DEBUG_CFLAGS' in os.environ:
        extra_compile_args = '\
            -std=c99 -pedantic -Wall -Wcast-align -Wcast-qual -Wpointer-arith \
            -Wchar-subscripts -Winline -Wnested-externs -Wbad-function-cast \
            -Wunreachable-code -Werror'
        define_macros = [('DEBUG', '1')]
    else:
        extra_compile_args = ['-O3']
        define_macros = []

    extensions = [Extension('ForcePy.Util', ['ForcePy/Util.%s' % ("pyx" if use_cython else "c")],
                            include_dirs=include_dirs ,
                            libraries = ['m'],
                            extra_compile_args=extra_compile_args),
                  Extension('ForcePy.NeighborList', ['ForcePy/NeighborList.%s' % ("pyx" if use_cython else "c")],
                            include_dirs=include_dirs ,
                            libraries = ['m'],
                            extra_compile_args=extra_compile_args),
                  Extension('ForcePy.Mesh', ['ForcePy/Mesh.%s' % ("pyx" if use_cython else "c")],
                            include_dirs=include_dirs ,
                            libraries = [],
                            extra_compile_args=extra_compile_args),

                  Extension('ForcePy.Basis', ['ForcePy/Basis.%s' % ("pyx" if use_cython else "c")],
                            include_dirs=include_dirs ,
                            libraries = [],
                            extra_compile_args=extra_compile_args)]


    setup(name              = 'ForcePy',
          version           = RELEASE,
          description       = 'Object-oriented Force Matching and coarse-graining toolkit',
          author            = 'Andrew White',
          author_email      = 'white.d.andrew@gmail.com',
          url               = 'http://github.com/whitead/ForcePy/',
          requires          = ['numpy (>=1.0.3)', 'biopython', 'MDAnalysis'],
          provides          = ['ForcePy'],
          license           = 'GPL 2',
          packages          = [ 'ForcePy'],
          package_dir       = {'ForcePy': 'ForcePy'},
          ext_modules       = extensions,
          classifiers       = CLASSIFIERS,
          long_description  = LONG_DESCRIPTION,
          cmdclass          = cmdclass,
          # all standard requirements are available through PyPi and
          # typically can be installed without difficulties through setuptools
          install_requires = ['numpy>=1.0.3',   # currently not useful because without numpy we don't get here
                              'MDAnalysis>=0.15' # Development version is currently required, so not good.
                              ],
          # extras can be difficult to install through setuptools and/or
          # you might prefer to use the version available through your
          # packaging system
          extras_require = {
                'analysis': ['matplotlib',
                             ],
                },
          zip_safe = False,     # as a zipped egg the *.so files are not found (at least in Ubuntu/Linux)
          )
