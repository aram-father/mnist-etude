from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mnistetude',
    version='0.0.1',
    description='MNIST data set classification using deep learning',
    long_description=readme,
    author='Wonseok Lee',
    author_email='aram_father@naver.com',
    url='https://github.com/aram-father/mnist-etude',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy', 'pickle', 'matplotlib'],
    python_requires='>=3',
    include_package_data=True
)
