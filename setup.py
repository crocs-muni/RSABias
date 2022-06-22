# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='rsabias',
    version='0.1',
    description='Tool to analyse RSA key generation and classification',
    long_description=readme,
    author='Matus Nemec',
    author_email='mnemec@mail.muni.cz',
    url='https://github.com/crocs-muni/RSABias',
    license=license,
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'rsabias=rsabias.cli:main'
        ]
    },
    install_requires=[
        'cycler==0.10.0',
        'gmpy2==2.0.8',
        'kiwisolver==1.2.0',
        'matplotlib==3.1.0',
        'mpmath==1.1.0',
        'numpy==1.22.0',
        'pandas==0.24.2',
        'pycryptodome==3.9.8',
        'pyparsing==2.4.7',
        'python-dateutil==2.8.1',
        'pytz==2020.1',
        'scipy==1.3.0',
        'seaborn==0.9.0',
        'six==1.15.0',
        'sympy==1.6.1',
        'xarray==0.12.1'
    ],
    include_package_data=True
)

