from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

VERSION = '1.0'
PACKAGE_NAME = 'dgpsi'
AUTHOR = 'Deyu Ming'
AUTHOR_EMAIL = 'deyu.ming.16@ucl.ac.uk'
URL = 'https://github.com/mingdeyu/DGP'

LICENSE = 'MIT'
DESCRIPTION = 'Deep Gaussian Process Emulation using Stochastic Imputation'
LONG_DESCRIPTION = (here / 'README.md').read_text(encoding='utf-8')

INSTALL_REQUIRES = [
      'numpy>=1.18.2',
      'numba>=0.51.2',
      'matplotlib>=3.2.1',
      'tqdm>=4.50.2',
      'scipy>=1.4.1',
      'scikit-learn>=0.22.0',
      'jupyter'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      author=AUTHOR,
      license=LICENSE,
      keywords = ["surrogate modelling","deep learning","stochastic EM","elliptical slice sampling"],
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      python_requires='>=3.7, <4',
      extras_require={'interactive': ['jupyter']},
      packages=find_packages(include=['dgpsi', 'dgpsi.*'])
      )