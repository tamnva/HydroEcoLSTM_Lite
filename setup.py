from setuptools import setup
from pathlib import Path

# Read the description
readme_file = Path(__file__).absolute().parent / "README.md"
with readme_file.open("r") as fp:
    long_description = fp.read()
    
setup(
    name='hydroecolstm_lite',
    version='0.1',    
    description='A python package for HydroEcological Modelling using LSTM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tamnva/HydroEcoLSTM_Lite.git',
    author='Tam V. Nguyen',
    author_email='tamnva@gmail.com',
    packages=['hydroecolstm_lite', 
              'hydroecolstm_lite.data',
              'hydroecolstm_lite.utility', 
              'hydroecolstm_lite.model',
              'hydroecolstm_lite.train'],
    python_requires='>=3.12',
    install_requires=['pandas',
                      'numpy',
                      'torch',
                      'PyYAML',
                      'pathlib'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],

    include_package_data=False,
)
