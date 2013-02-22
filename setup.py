import os
import setuptools

setuptools.setup(
    name='lmj.sim',
    version='0.0.1',
    namespace_packages=['lmj'],
    packages=setuptools.find_packages(),
    author='Leif Johnson',
    author_email='leif@leifjohnson.net',
    description='Yet another OpenGL-with-physics simulation framework',
    long_description=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='http://github.com/lmjohns3/py-sim/',
    keywords=('simulation '
              'physics '
              'ode '
              'visualization '
              ),
    install_requires=['numpy', 'glumpy', 'PyOpenGL', 'plac', 'PyODE'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
