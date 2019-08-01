import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='rlmodels',  

     version='1.0.5',

     author="Nestor Sanchez",

     author_email="nestor.sag@gmail.com",

     packages = setuptools.find_namespace_packages(include=['rlmodels.*']),

     description="Implementation of some popular reinforcement learning models",

     license = "MIT",

     install_requires=[
        'torch>=1.1.0',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',

    ],

     long_description=long_description,

     long_description_content_type="text/markdown",

     url="https://github.com/nestorsag/rlmodels",

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

     download_url = 'https://github.com/nestorSag/rlmodels/archive/1.0.5.tar.gz'

 )