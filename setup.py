from setuptools import find_packages, setup
from typing import List

def get_requirements(filePath:str)->List[str]:
    '''This function returns list of requirements for the project'''
    requirements=[]
    with open(filePath) as fileObj:
        requirements = fileObj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='EMart',
    version='0.0.1',
    author='Arjun Gadvi',
    author_email='arjun.gadvi@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)