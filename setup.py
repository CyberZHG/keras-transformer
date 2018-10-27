from setuptools import setup, find_packages

setup(
    name='keras-transformer',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/CyberZHG/keras-transformer',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='Transformer implemented in Keras',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'keras',
        'keras-multi-head==0.8.0',
        'keras-layer-normalization=0.2.0',
        'keras-position-wise-feed-forward=0.1.0',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
