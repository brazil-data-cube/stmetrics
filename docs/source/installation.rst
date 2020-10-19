Installation
==============

.. _windows:

---------------------
 stmetrics on Windows
---------------------

**stmetrics** is supported under Windows with the following caveats:

- Python 3.5 or higher
- MSVC compiler is not supported.

**stmetrics** requires a working C++ compiler, due to the spatial-temporal segmentation algorithm available. Configuring such a compiler is the critical step in getting the spatial-temporal algorithm running.

The package is tested against the mingw-w64 compiler which works on both Python versions (3.x)
and supports x86 and x64.


Installing Python
-----------------

The following instructions assume you have installed Python as packaged in the Anaconda
Python distribution.

Open Command prompt
-------------------

With `Anaconda Prompt` update conda package manager to the latest version::

    conda update conda

Create a conda virtual environment (recomended)
-----------------------------------------------

To do it, use::

    conda create -n new_env python=3.7

where ``new_env`` is the name of the environment.

After this activate environment with::

    conda activate new_env


Installing C++ compiler
-----------------------

As previously stated, the package requires the mingw-w64 compiler. To install mingw-w64 compiler type::

    conda install libpython m2w64-toolchain -c msys2

This will install

- ``libpython`` package which is needed to import mingw-w64. <https://anaconda.org/anaconda/libpython>
- mingw-w64 toolchain. <https://anaconda.org/msys2/m2w64-toolchain>

.. Hint::

    ``libpython`` setups automatically ``distutils.cfg`` file, but if that is failed
    use the following instructions to setup it manually. Go to enviroment Lib path in ``Anaconda3\\envs\\new_env\\Lib\\distutils`` and create a ``distutils.cfg`` file with a text editor (e.g. Notepad) and add the following lines::

        [build]
        compiler=mingw32

    To find the correct ``distutils`` path, run the following lines in ``python``:

        >>> import distutils
        >>> print(distutils.__file__)

Install dtaidistance package
----------------------------

The dtaidistance package is a key factor of stmetrics. However, due to some issues, windows users need to compiled and install directly from source. This was tested with version 1.2.2 and 1.2.4. 

.. Caution::
    **Make sure that you have numpy and cython already installed!**

* Download the source from :download:`https://github.com/wannesm/dtaidistance<static/https://github.com/wannesm/dtaidistance>`
* Compile the C extensions: ``python setup.py build_ext --inplace``
* Install into your site-package directory: ``python setup.py install``

.. Caution::
    If OpenMP is not available at your system use::

        python3 setup.py --noopenmp build_ext --inplace

.. WARNING::
    We strongly advise install OpenMP to use spatial-temporal segmentation algorithm.

.. Hint::
    If after installation fast computation using DTW be not available, try to follow the steps from this page:
    https://dtaidistance.readthedocs.io/en/latest/usage/installation.html#from-pypi

Install proper 3rd-party packages
---------------------------------

Before installing the package make sure that you have correctly installed Shapely and Rasterio packages.

Using conda do this::

    conda config --add channels conda-forge

    conda install shapely

    conda install rasterio

    conda install geopandas

Installing stmetrics
--------------------

You can pip install it straight from git::

	pip install git+https://github.com/brazil-data-cube/stmetrics

