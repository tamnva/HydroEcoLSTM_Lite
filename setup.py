from setuptools import setup, find_packages
from pathlib import Path

# Read the description
readme_file = Path(__file__).absolute().parent / "README.md"
with readme_file.open("r") as fp:
    long_description = fp.read()
    
setup(
    name='hydroecolstm_lite',
    version='0.1.0',
    description='A Python package for hydro-ecological modelling using LSTM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tamnva/HydroEcoLSTM_Lite.git',
    author='Tam V. Nguyen',
    author_email='tamnva@gmail.com',
    packages=find_packages(exclude=("tests", "examples")),
    python_requires='>=3.8',
    install_requires=[
        'pandas>=1.0',
        'numpy>=1.20',
        'torch',
        'PyYAML',
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],

    include_package_data=False,
    license='GPLv3',
    keywords=['hydrology', 'ecology', 'lstm', 'deep-learning', 'time-series'],
)
