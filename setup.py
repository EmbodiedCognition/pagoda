import os
import setuptools

setuptools.setup(
    name='pagoda',
    version='0.1.0',
    packages=setuptools.find_packages(),
    author='Leif Johnson',
    author_email='leif@cs.utexas.edu',
    description='yet another OpenGL-with-physics simulation framework',
    long_description=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='http://github.com/EmbodiedCognition/py-sim/',
    keywords=('simulation '
              'physics '
              'ode '
              'visualization '
              ),
    install_requires=['climate', 'numpy', 'pyglet'],
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
