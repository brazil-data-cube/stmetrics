Installation
==============

.. _windows:

---------------------
 stmetrics on Windows
---------------------

stmetrics is partially supported under Windows with the following caveats:

- Python 2.7: Doesn't support parallel sampling. When drawing samples ``n_jobs=1`` must be used)
- Python 3.5 or higher: Parallel sampling is supported
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

For windows users:
------------------

Please before installing the package make sure that you have correctly installed the Shapely package.

Using conda do this::

    conda config --add channels conda-forge

    conda install shapely

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

Installing stmetrics
--------------------

You can pip install it straight from git:

With pip::

	pip install stmetrics

