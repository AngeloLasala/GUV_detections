from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='guv_detection',
    version='0.0',
    description='YOLO for GUV detection',
    author='Angelo Lasala',
    author_email='Lasala.Angelo@santannapisa.it',
    packages=find_packages(),
    install_requires=requirements,
)