.. role:: python(code)
    :language: python

.. SKA Science Data Challenge Solution documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to SKA Science Data Challenge 1 Solution's documentation!
=================================================================

This package is an open source set of tools aimed at solving the SKA Science Data Challenge 1. This has been designed to make it as easy as possible to reproducibly develop and run a solution to the challenge in an automated or interactive manner via a simple Python API and leveraging containerised computing environments.

There are broadly 3 steps to producing a result that can be scored for the challenge:

1) Image preparation; to perform the primary beam correction and cut out a section of the image which represents the training area (the source list for which is known).
2) Source finding; this is a wrapper around the PyBDSF (Python blob detection source finder) algorithm which is designed to automatically identify sources in radio astronomical image data.
3) Source classification; to identify the most likely class (out of star forming galaxy, steep-spectrum AGN and flat-spectrum AGN) for each detected source.

The resulting catalogues can then be converted into the required format for the SDC1 scoring algorithm to run on, allowing for the score calculation.

.. toctree::
   :maxdepth: 2
   :caption: SDC1 Solution

   sdc1_image
   source_finder

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
