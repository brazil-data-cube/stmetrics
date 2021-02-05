.. _depend-ref:

Dependencies
=============

Installing Python
-----------------

The following instructions assume you have installed Python as packaged in the Anaconda Python distribution. The stmetrics in different enviroments using Anaconda.

Open Command prompt
-------------------

With `Anaconda Prompt` update conda package manager to the latest version::

    conda update conda

Create a conda virtual environment (recommended)
------------------------------------------------

To do it, use::

    conda create -n new_env python=3.7

where ``new_env`` is the name of the environment.

After this, activate the environment with::

    conda activate new_env


Install proper 3rd-party packages
---------------------------------

Before installing **smetrics** make sure that you have correctly installed Shapely, Rasterio and Geopandas.

To install using conda, please use::

    conda config --add channels conda-forge

    conda install shapely

    conda install rasterio

    conda install geopandas


**stmetrics** dependecies on Windows
------------------------------------

-----------------------
Installing C++ compiler
-----------------------

As previously stated, the package requires the mingw-w64 compiler. To install mingw-w64 compiler type::

    conda install libpython m2w64-toolchain -c msys2

This will install

- ``libpython`` package which is needed to import mingw-w64. <https://anaconda.org/anaconda/libpython>
- mingw-w64 toolchain. <https://anaconda.org/msys2/m2w64-toolchain>

.. Hint::

    ``libpython`` creates automatically ``distutils.cfg`` file, but if it failed use the following instructions to setup it manually. Go to environment Lib path in ``Anaconda3\\envs\\new_env\\Lib\\distutils`` and create a ``distutils.cfg`` file with a text editor (e.g. Notepad) and add the following lines::

        [build]
        compiler=mingw32

    To find the correct ``distutils`` path, run the following lines in ``python``:

        >>> import distutils
        >>> print(distutils.__file__)

----------------------------
Install dtaidistance package
----------------------------

The dtaidistance package is mandatory for stmetrics. However, due to some issues, Windows users need to compile and install directly from source. This was tested with version 1.2.4.

.. Caution::
    **Make sure that you have numpy and cython already installed!**

* Download the source from https://github.com/wannesm/dtaidistance
* Compile the C extensions: ``python setup.py build_ext --inplace``
* Install into your site-package directory: ``python setup.py install``

.. Caution::
    If OpenMP is not installed in your system you can use::

        python3 setup.py --noopenmp build_ext --inplace

    However, if it is not installed, make sure the C++ compiler was properly installed. 

.. Hint::
    If after installation fast computation using DTW be not available, follow the steps from this page:
    https://dtaidistance.readthedocs.io/en/latest/usage/installation.html#from-pypi

stmetrics on Linux
---------------------

At this moment we don't have reports on issues regarding ubuntu installation.

Follow the installation for 3rd paty dependecies as describe above. 

For dtaidistance package just make sure your compiler has openmp and use::

	pip install dtaidistance[numpy]