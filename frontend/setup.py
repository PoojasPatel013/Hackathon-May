# frontend/setup.py
from setuptools import setup, find_packages

setup(
    name='disaster-prediction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'torch',
        'numpy',
        'scikit-learn',
        'pandas',
        'pyyaml',
        'plotly',
        'geopandas'
    ]
)