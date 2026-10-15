from setuptools import setup

setup(
   name='pstfd_project',
   version='1.0',
   description='Module Developed During PSTFD Exercise Course',
   author='Trong Tin Nguyen',
   author_email='trongtin.nguyen@tum.de',
   packages=['pstfd_project'],  #same as name
   install_requires=['numpy', 'matplotlib', 'h5py', 'tensorflow', 'scipy', 'pathlib', 'jax', 'jaxlib', 'optax', 'colorednoise'], #external packages as dependencies
)