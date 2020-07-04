Installation
==============

.. _windows:

---------------------
 stmetrics on Windows
---------------------

stmetrics is partially supported under Windows with the following caveats:

- Python 3.5 or higher
- MSVC compiler is not supported.

stmetrics requires a working C++ compiler, due to the spatial-temporal segmentation algorithm available. Configuring such a compiler is the critical step in getting stmetrics running.

stmetrics is tested against the mingw-w64 compiler which works on both Python versions (3.x)
and supports x86 and x64.


Installing Python
-----------------

There several ways of installing stmetrics on Windows. The following instructions
assume you have installed Python as packaged in the `Anaconda
Python distribution <https://www.anaconda.com/download/#windows>`_
or `Miniconda distribution <https://conda.io/miniconda.html>`_.

Open Command prompt
-------------------

All the following commands are written in a command line prompt. You can use one like
`Anaconda Prompt` if you installed Python using Anaconda in the previous step, or
`Command Prompt` which comes with Windows.

Test that the conda package manager is properly installed by typing::

    conda info

To update conda package manager to the latest version::

    conda update conda

Create a conda virtual environment (optional)
---------------------------------------------

It is a good practice to keep specific projects on their on aspecific conda virtual environments. To do it, use::

    conda create -n new_env python=3.7

where ``new_env`` is the name of the environment.

After this activate environment with::

    conda activate new_env

or if your conda doesn't include ``conda activate`` use::

    activate new_env

To close the environment type::

    deactivate

Installing C++ compiler
-----------------------

To install mingw-w64 compiler toolchain with ``conda`` package manager which comes with the Anaconda package, we recommend the following steps.

To install mingw-w64 compiler type::

    conda install libpython m2w64-toolchain -c msys2

This will install

- ``libpython`` package which is needed to import mingw-w64. <https://anaconda.org/anaconda/libpython>
- mingw-w64 toolchain. <https://anaconda.org/msys2/m2w64-toolchain>

``libpython`` setups automatically ``distutils.cfg`` file, but if that is failed
use the following instructions to setup it manually

In ``PYTHONPATH\\Lib\\distutils`` create a ``distutils.cfg`` file with a text editor (e.g. Notepad) and add the following lines::

    [build]
    compiler=mingw32

To find the correct ``distutils`` path, run the following lines in ``python``::

    >>> import distutils
    >>> print(distutils.__file__)

Install dtaidistance package
----------------------------

The dtaidistance package is a key factor of stmetrics. However, due to some issues, windows users need to compiled and install directly from source.

* Download the source from https://github.com/wannesm/dtaidistance
* Compile the C extensions: ``python3 setup.py build_ext --inplace``
* Install into your site-package directory: ``python3 setup.py install``

This requires OpenMP to be available on your system. If this is not the case, use:

::

    $ python3 setup.py --noopenmp build_ext --inplace

Before installing the package make sure that you have correctly installed Shapely and Rasterio packages.

Using conda do this::

    conda config --add channels conda-forge

    conda install shapely

    conda install rasterio

Installing stmetrics
--------------------

You can pip install it straight from git:

With pip::

	pip install git+https://github.com/brazil-data-cube/stmetrics

