.. stmetrics documentation master file, created by
   sphinx-quickstart on Tue Feb  4 13:53:12 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to stmetrics's documentation!
=====================================


.. toctree::
   :caption: Introduction
   :maxdepth: 0
   :hidden:

   about.rst
   license.rst
   contributors.rst
   publications.rst


.. toctree::
   :caption: Package Components
   :maxdepth: 2
   :hidden:

   metrics.rst
   polar.rst
   basics.rst
   fractal.rst
   spatial.rst
   utils.rst


.. toctree::
   :caption: Setup
   :maxdepth: 2
   :hidden:

   dependencies.rst
   installation.rst

   
.. toctree::
   :caption: Tutorials
   :maxdepth: 2
   :numbered:
   :hidden:

   examples/TimeMetrics
   examples/Benchmark


.. Note::
   This documentation is not finished. Parts of the description are \
   incomplete and may need corrections. Please, come back later for \
   the definitive documentation.

**stmetrics** is a python package that aims at making the process of feature \
extraction of state-of-the-art time-series as simple as possible.

It provides functions to support time-series analyzes that includes not only \
the feature extraction but also spatio-temporal analysis through a \
spatio-temporal segmentation and filtering approaches that are a first step \
to the full exploitation of the spatio-temporal information.

The documentation presented here summarize the technical aspects of the \
package. The methodoogical application of the package and it's methods are \
available at the :ref:`publications-ref` section.