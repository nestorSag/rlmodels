import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='rlmodels',  

     version='1.0.1',

     author="Nestor Sanchez",

     author_email="nestor.sag@gmail.com",

     description="Implementation of some popular reinforcement learning models",

     license = "MIT",

     install_requires=[
        'torch>=1.1.0',
        'numpy>=1.16.4',
        'pandas>=0.25.0',
        'matplotlib>=3.1.1',
        'seaborn>=0.9.0',

    ],

     long_description=long_description,

     long_description_content_type="text/markdown",

     url="https://github.com/nestorsag/rlmodels",

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

     download_url = 'https://github.com/nestorSag/rlmodels/archive/1.0.1.tar.gz'

 )