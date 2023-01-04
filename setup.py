

#https://docs.python.org/2/distutils/setupscript.html

from setuptools import setup
from distutils.core import Extension
import sys
import sysconfig
import os, os.path
import subprocess
import glob

# just in case ...
#https://github.com/blog/2104-working-with-submodules

#accumulate C extension
accumulate_c_src= glob.glob('exptool/basis/accumulate_c/*.c')

# eventually put numpy header for arrays in here
accumulate_include_dirs= ['exptool/basis/accumulate_c']

accumulate_c= Extension('exptool.basis._accumulate_c',
                             sources=accumulate_c_src,
                             #libraries=pot_libraries,
                             include_dirs=accumulate_include_dirs)
                             #extra_compile_args=extra_compile_args,
                             #extra_link_args=extra_link_args)

ext_modules = []
ext_modules.append(accumulate_c)


setup(name='exptool',
      version='0.3.1',
      description='exp analysis in Python',
      author='Michael Petersen',
      author_email='michael.petersen@roe.ac.uk,petersen@iap.fr',
      license='New BSD',
      #long_description=long_description,
      url='https://github.com/michael-petersen/exptool',
      package_dir = {'exptool/': ''},
      packages=['exptool','exptool/analysis','exptool/basis',
                'exptool/io','exptool/orbits','exptool/utils','exptool/observables','exptool/models','exptool/tests'],
      package_data={'': ['README.md','LICENSE'],'exptool/tests':['*.dat'],'exptool/tests/data':['*.dat']},
      include_package_data=True,
      install_requires=['numpy>=1.7','scipy','matplotlib'],
      ext_modules=ext_modules,
      classifiers = ["Programming Language :: Python", "Intended Audience :: Science/Research"]
      )
