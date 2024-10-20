#use to create a package for my machine learning model so that it can be use later.

from setuptools import find_packages, setup
from typing import List

HYPHOM_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHOM_E_DOT in requirements:
            requirements.remove(HYPHOM_E_DOT)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Sanchit',
    author_email='atcsanchit@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")

)