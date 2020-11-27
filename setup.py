from setuptools import setup

setup(
   name='mtn_mmsr',
   version='1.0',
   description='solve mmsr',
   author='Shashank',
   author_email='f20180443@goa.bits-pilani.ac.in',
   packages=setuptools.find_packages(),
   install_requires=['numpy', 'scipy','torch','numdifftools'], #external packages as dependencies
)