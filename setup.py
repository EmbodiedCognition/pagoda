import os
import setuptools

README = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.rst')

setuptools.setup(
    name='pagoda',
    version='0.1.0',
    packages=setuptools.find_packages(),
    package_data={'': ['*.peg']},
    author='UT Vision, Cognition, and Action Lab',
    author_email='leif@cs.utexas.edu',
    description='pyglet + ode + numpy: a simulation framework',
    long_description=open(README).read(),
    license='MIT',
    url='http://github.com/EmbodiedCognition/pagoda/',
    keywords=('simulation '
              'physics '
              'ode '
              'visualization '
              ),
    install_requires=['click', 'numpy', 'parsimonious', 'pyglet'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        ],
    )
