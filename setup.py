from setuptools import setup, find_packages


def get_requirements():
    with open('requirements.txt', 'r') as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))

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
    install_requires=get_requirements(),
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
